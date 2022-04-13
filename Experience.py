
class Experience():
  def __init__(self, state, action, reward, next_state, done, label):
    self.state = state
    self.action = action
    self.reward = reward
    self.next_state = next_state
    self.done = done
    self.label = label
  
  def get_experience(self):
    return self.state.shape, self.action, self.reward, self.next_state.shape, self.done

  def get_label(self):
    return self.label