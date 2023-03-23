import gen_embs as gen_embs

class DatabaseSimulator():

    def __init__(self, init_user = True):
        self.users_emb = {}
        if init_user:
            self._init_users()

    def insert_user(self, user_name, user_emb):
        self.users_emb[user_name] = user_emb
        self._init_users()

    def get_emb(self, user_name):
        if  self.users_emb.get(user_name) is not None:
            return self.users_emb[user_name]
        else:
            raise ValueError("User %s not found" % user_name)
        
    def update_emb(self, user_name, user_emb):
        if self.user_emb.get(user_name) is not None:
            self.users_emb[user_name] = user_emb
        else:
            raise ValueError("User %s not found" % user_name)
        
    def _init_users(self):
        self.users_emb = gen_embs.get_all_embds(read_from_file=True)
        