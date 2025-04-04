import numpy as np
import pickle
from connection import connect, get_state_reward

# Constantes do jogo
NUM_PLATAFORMAS = 24
NUM_DIRECOES = 4
ACTIONS = ["left", "right", "jump"]
NUM_ACTIONS = len(ACTIONS)

# Parâmetros do Q-Learning
ALPHA = 0.3       # Taxa de aprendizado
GAMMA = 0.7       # Valorização de recompensas futuras
EPSILON = 0.5     # Exploração 
EPSILON_MIN = 0.01 # Mínimo de exploração
EPISODES = 200    # episódios para aprendizado completo


class QLearningAgent:
    def __init__(self):
        self.q_table = np.loadtxt("qtable.txt")
        
    def state_to_index(self, state_bin):
        """
        Converte o estado binário recebido do jogo em um índice para a Q-table
        Formato do estado: 7 bits (5 para plataforma + 2 para direção)
        """
        platform = int(state_bin[2:7], 2)  # Primeiros 5 bits representam a plataforma (0-23)
        direction = int(state_bin[7:], 2)  # Últimos 2 bits representam a direção (0-3)
        return platform * NUM_DIRECOES + direction
    
    def choose_action(self, state_index, epsilon):
        
        values = self.q_table[state_index]
        if np.std(values) < 0.1:  # Valores muito próximos
            return np.random.randint(NUM_ACTIONS)
        return np.argmax(values) if np.random.random() > epsilon else np.random.randint(NUM_ACTIONS)
    def update_q_table(self, state_index, action, reward, new_state_index):
      

        best_next_action = np.argmax(self.q_table[new_state_index])
        td_target = reward + GAMMA * self.q_table[new_state_index][best_next_action]
        td_error = td_target - self.q_table[state_index][action]
        self.q_table[state_index][action] += ALPHA * td_error
    
    def save_q_table(self, filename):
        """
        Salva a Q-table em um arquivo de texto:
        - Apenas dados numéricos
        - Valores separados por espaço
        - Uma linha por estado
        - Ordem das colunas: [left, right, jump]
        """
        # Garante que os números serão formatados com ponto decimal e sem notação científica
        np.savetxt(filename, self.q_table, fmt='%.6f', delimiter=' ')

def train_agent():
    """
    Função principal para treinar o agente
    """
    # Conecta ao jogo
    socket = connect(2040)  # Ajuste a porta conforme necessário
    
    # Inicializa o agente
    agent = QLearningAgent()
    
    # Loop de treinamento
    for episode in range(EPISODES):
        
        # Obtém o estado inicial
        state_bin, reward = get_state_reward(socket, "jump")  # Ação dummy para iniciar
        state_index = agent.state_to_index(state_bin)
        done = False
        total_reward = 0

        current_epsilon = max(EPSILON_MIN, EPSILON - ((episode//100)*0.1))

        while not done:
            # Escolhe uma ação
            action_idx = agent.choose_action(state_index, current_epsilon)
            action = ACTIONS[action_idx]

            # Executa a ação e obtém novo estado e recompensa
            new_state_bin, reward = get_state_reward(socket, action)
            new_state_index = agent.state_to_index(new_state_bin)  
                
            if reward == -100:
                done = True
          
            total_reward += reward
            # Atualiza a Q-table
            agent.update_q_table(state_index, action_idx, reward, new_state_index)
            
            # Atualiza o estado atual
            state_index = new_state_index



        agent.save_q_table("qtable.txt")    
        print(f"Episódio: {episode + 1}, Recompensa Total: {total_reward}, Current Epsilon: {current_epsilon}")
    

    print("Treinamento concluído")
    
    return agent

def run_trained_agent(q_table_file, agent):
    # Carrega a Q-table
    q_table = np.loadtxt(q_table_file, delimiter=' ')
    
    # Conecta ao jogo
    socket = connect(2040)  # Ajuste a porta conforme necessário
    
    state_bin, reward = get_state_reward(socket, "jump")  # Ação dummy para iniciar
    state_index = agent.state_to_index(state_bin)
    
    done = False
    contador = 20
    while not done:
        # Sempre escolhe a melhor ação (epsilon = 0)
        action_idx = np.argmax(q_table[state_index])
        action = ACTIONS[action_idx]

        
        new_state_bin, reward = get_state_reward(socket, action)
        state_index = agent.state_to_index(new_state_bin)
        
        if reward == -100 or contador == 1:
            done = True
        contador -= 1


if __name__ == "__main__":
    # Treina o agente
    agent = train_agent()
    
    # Executa o agente treinado
    
    agente_run = QLearningAgent()
    run_trained_agent("qtable.txt", agente_run)
