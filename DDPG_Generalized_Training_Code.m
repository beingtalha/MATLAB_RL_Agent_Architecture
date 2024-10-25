%% DDPG Model Network and Code
%% Flags / Settings
parallelComputing_flag = 0;  % Whether use Parallel computing
load_Saved_Agent_flag = 0;
%% Load Saved Agent
if load_Saved_Agent_flag == 1
    savedAgent_dir = 'saved_Agents01';   
    listing = dir(fullfile(savedAgent_dir, '*.mat'));
    for i = 1:length(listing)
         temp_String= string(listing(i).name);
         temp_String = extractAfter(temp_String,5); 
         temp_String = extractBefore(temp_String,'.mat'); 
         agent_names(i,1) = str2num(temp_String);
         
    end
    sorted_agent_names = sort(agent_names,'ascend');
    last_Agent = sorted_agent_names(end);
    agent_Name = append('\Agent',num2str(last_Agent), '.mat');
    load([savedAgent_dir agent_Name]);
    [ep_reward ep_no] = max(savedAgentResult.EpisodeReward);
    load([savedAgent_dir append('\Agent', num2str(ep_no), '.mat')]);
    plot(savedAgentResult)
end
%% Model Intialization
mdl = 'Your Simulink Mdl Name';
open_system(mdl); % opens system model
agentblk = [mdl 'Location/To/RL Agent Block']; % Replace with the location to RL Agent Block in Simulink

%% Sample Time & Simulation Duration
T_Sample = 0.1; % Replace it with your own sample time
T_Total = 205; % Replace it with with your own Total Simulation Time
set_param(mdl,'StartTime','0','StopTime',int2str(T_Total)); % Set Start and Stop Time in Simulink

%% Observation Info
numObs = 5; % Enter Observation Number
obsInfo = rlNumericSpec([numObs 1],'LowerLimit',(-inf*zeros(numObs,1)),'UpperLimit',inf*zeros(numObs,1)); % Upper and Lower Limit of +- inf
obsInfo.Name = 'Observations';
obsInfo.Description = 'This is the description about observation info';
numOfObservations = obsInfo.Dimension(1); 

%% Action Info

Act1_Min = -5;
Act1_Max = 5;

% theta Controller
Act2_Min = -0.04;
Act2_Max = 0.04;

% Action Object
numAct = 1; % Number of Actions
actInfo = rlNumericSpec([numAct 1],'LowerLimit',Act1_Min*zeros(numAct,1),'UpperLimit' ,Act2_Min*zeros(numAct,1));
actInfo.Name = 'Action';
numActions = actInfo.Dimension(1);

%% Create Environment
%rl_env = rlSimulinkEnv(mdl, agentblk, obsInfo,
%actInfo,'UseFastRestart','on'); % if you want to use Fast Restart
rl_env = rlSimulinkEnv(mdl, agentblk, obsInfo, actInfo); % Creates Environment
%% Environment Reset Function
% To define the initial condition for the certain variable, specify an environment reset function using an anonymous function handle. 
% The reset function localResetFcn, which is defined at the end of the example.
rl_env.ResetFcn = @(in)localResetFcn(in);
% Fix the random generator seed for reproducibility.
rng('default')
%% Create Agent DDPG
% A DDPG agent approximates the long-term reward given observations and actions using a critic value function representation. 
% To create the critic, first create a deep neural network with two inputs, the state and action, and one output. 
% For more information on creating a neural network value function representation, see Create Policy and Value Function Representations.
nI = obsInfo.Dimension(1);  % number of inputs
nL = 128;                            % number of neurons
nO = actInfo.Dimension(1);    % number of outputs

statePath = [
    featureInputLayer(nI,'Normalization','none','Name','observation')
    fullyConnectedLayer(nL,'Name','fc1')
    reluLayer('Name','relu1')
    fullyConnectedLayer(nL,'Name','fc2')
    additionLayer(2,'Name','add')
    reluLayer('Name','relu2')
    fullyConnectedLayer(nL,'Name','fc3')
    reluLayer('Name','relu3')
    fullyConnectedLayer(1,'Name','fc4')];

actionPath = [
    featureInputLayer(nO,'Normalization','none','Name','action')
    fullyConnectedLayer(nL, 'Name', 'fc5')];

%% Critic Netwrok
criticNetwork = layerGraph(statePath);
criticNetwork = addLayers(criticNetwork, actionPath);
    
criticNetwork = connectLayers(criticNetwork,'fc5','add/in2');

% Specify options for the critic representation using rlRepresentationOptions.
criticOptions = rlRepresentationOptions('Optimizer','adam','LearnRate',1e-3,'GradientThreshold',1,'L2RegularizationFactor',2e-4,'UseDevice',"gpu"); %Use GPU for Training

% Create the critic representation using the specified neural network and options. 
% You must also specify the action and observation info for the critic, which you obtain from the environment interface. 
% For more information, see rlQValueRepresentation.
critic = rlQValueRepresentation(criticNetwork,obsInfo,actInfo,'Observation',{'observation'},'Action',{'action'},criticOptions);

% Design, visualize, and train deep learning networks
%deepNetworkDesigner(criticNetwork);
% View the critic network configuration.
figure('Name','Critic Network');
plot(criticNetwork);

%% Actor Netwrok
% A DDPG agent decides which action to take given observations by using an actor representation.
% To create the actor, first create a deep neural network with one input, the observation, and one output, the action.
% Construct the actor similarly to the critic. For more information, see rlDeterministicActorRepresentation.
actorNetwork = [
    featureInputLayer(nI,'Normalization','none','Name','observation')
    fullyConnectedLayer(nL,'Name','fc1')
    reluLayer('Name','relu1')
    fullyConnectedLayer(nL,'Name','fc2')
    reluLayer('Name','relu2')
    fullyConnectedLayer(nL,'Name','fc3')
    reluLayer('Name','relu3')
    fullyConnectedLayer(1,'Name','fc4')
    tanhLayer('Name','tanh1')
    scalingLayer('Name','ActorScaling1','Scale',Act1_Max,'Bias',-0.5)];

actorOptions = rlRepresentationOptions('Optimizer','adam','LearnRate',1e-4,'GradientThreshold',1,'L2RegularizationFactor',1e-5,'UseDevice',"gpu"); %Use GPU for Training

actor = rlDeterministicActorRepresentation(actorNetwork,obsInfo,actInfo,'Observation',{'observation'},'Action',{'ActorScaling1'},actorOptions);

% Design, visualize, and train deep learning networks
%deepNetworkDesigner(actorNetwork);
% View the actor network configuration.
figure('Name','Actor Network');
plot(layerGraph(actorNetwork));

%% Agent Options
% To create the DDPG agent, first specify the DDPG agent options using rlDDPGAgentOptions.
agentOptions = rlDDPGAgentOptions(...
    'SampleTime',T_Sample,...
    'TargetSmoothFactor',1e-3,...
    'SaveExperienceBufferWithAgent',true, ...
    'ExperienceBufferLength',1e8,...
    'DiscountFactor',0.99,...
    'MiniBatchSize',64);
agentOptions.NoiseOptions.Variance = 0.3;
agentOptions.NoiseOptions.VarianceDecayRate = 1e-5;
agentOptions.ResetExperienceBufferBeforeTraining = false;
agentOptions.SaveExperienceBufferWithAgent = true;

% Then, create the DDPG agent using the specified actor representation, critic representation, and agent options. 
% For more information, see rlDDPGAgent.
agent = rlDDPGAgent(actor,critic,agentOptions);

%% Specify Training Options and Train Agent
% For this example, the training options for the DDPG and TD3 agents are the same.
% Run each training session for 5000 episodes with each episode lasting at most maxSteps time steps.
% Display the training progress in the Episode Manager dialog box (set the Plots option) and disable the command line display (set the Verbose option).
% Terminate the training only when it reaches the maximum number of episodes (maxEpisodes). Doing so allows the comparison of the learning curves for multiple agents over the entire training session. 
maxEpisodes = 1000000;
maxSteps = floor(T_Total/T_Sample);
trainingOptions = rlTrainingOptions(...
    'MaxEpisodes',maxEpisodes,...
    'MaxStepsPerEpisode',maxSteps,...
    'ScoreAveragingWindowLength',100,...
    'Verbose',true,...
    'Plots','training-progress',...
    'StopTrainingCriteria','EpisodeCount',...
    'StopTrainingValue',maxEpisodes,...
    'SaveAgentCriteria','EpisodeFrequency',...
    'SaveAgentValue',100, ...
    'SaveAgentDirectory','saved_Agents01' ...
    ); % saves every 100th episode for DDPG Agent

% To train the agent in parallel, specify the following training options. 
% Training in parallel requires Parallel Computing Toolbox™. 
% If you do not have Parallel Computing Toolbox software installed, set UseParallel to false.
% Set the UseParallel option to true.
% Train the agent in parallel asynchronously.
% After every 32 steps, have each worker send experiences to the parallel pool client (the MATLAB® process which starts the training). 
% DDPG and TD3 agents require workers to send experiences to the client.
if parallelComputing_flag==1
     save_system(mdl);
     num_cores = feature('numcores'); % Get number of CPU Cores
     parpool(floor(num_cores*.75)); % Use 75% fo Available Cores
     trainingOptions.UseParallel = true;
     trainingOptions.ParallelizationOptions.Mode = 'async';
     trainingOptions.ParallelizationOptions.StepsUntilDataIsSent = 32;
     trainingOptions.ParallelizationOptions.DataToSendFromWorkers = 'Experiences';
 end

%% Train the agent.
trainingStats = train(agent,rl_env,trainingOptions)

%% Simulate DDPG Agent
% To validate the performance of the trained agent, simulate the agent within the Simulink environment by uncommenting the following commands. 
% For more information on agent simulation, see rlSimulationOptions and sim.

 simOptions = rlSimulationOptions('MaxSteps',maxsteps);
 experience = sim(env,agent,simOptions);

%% Reset Function Definition
function in = localResetFcn(in)
    mdl = 'Your Simulink Model Name';    
    in =Simulink.SimulationInput(mdl); 
    
    % LOGIC TO INITIALIZE A VARIABLE HERE
    % alt = answer;

    %change  value in model worspace
    mdlWks = get_param(mdl,'ModelWorkspace');
    assignin(mdlWks,'variable name',alt) % assigns value to Base Workspace of Model
end

%% CopyRights
% Everything is designed with the Help of Mathworks and its documentation. 
% Talha Bin Riaz