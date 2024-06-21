class ContextManager:
    def __init__(self):
        self.context = {}

    def update_context(self, user_id, key, value):
        if user_id not in self.context:
            self.context[user_id] = {}
        self.context[user_id][key] = value

    def get_context(self, user_id, key):
        return self.context.get(user_id, {}).get(key, None)

# Example usage
context_manager = ContextManager()
context_manager.update_context('user123', 'last_intent', 'greeting')
#print(context_manager.get_context('user123', 'last_intent'))
