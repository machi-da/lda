# -*- coding:utf-8 -*-
import re
import numpy as np
from nltk.stem import WordNetLemmatizer

class LDA:
    # トピックス数:Z, ハイパーパラメータ:alpha,beta
    def __init__(self, Z, alpha=0.3, beta=0.3):
        self.Z = Z
        self.alpha = alpha
        self.beta = beta
        
    def set_initialize(self, file_name):
        self.corpus = []
        self.voc_dic = {}
        self.topic = []
        
        self.create_corpus(file_name)
        
        self.n_wz = np.zeros((self.Z, len(self.voc_dic))) + self.beta # 単語wがトピックzに現れた回数をカウント
        self.n_dz = np.zeros((len(self.corpus), self.Z)) + self.alpha # 文書dでの単語のトピックをカウント
        self.n_z = np.zeros(self.Z) + len(self.voc_dic) * self.beta # トピックごとの文書数をカウント
        self.n_d = np.zeros(len(self.corpus)) + self.Z * self.alpha # 各文書の単語数をカウント(正確には各文書の単語のトピック数を合計したもの)
        
        for i, m in enumerate(self.corpus):
            z_rand = np.random.randint(0, self.Z, len(m))
            self.topic.append(z_rand) # トピックをランダムに決める
            self.n_d[i] += len(m)
            for n, z in zip(m, z_rand):
                self.n_dz[i, z] += 1
                self.n_wz[z, self.voc_dic[n]] += 1
                self.n_z[z] += 1

    # コーパスと単語のインデックス辞書の作成
    def create_corpus(self, file_name):
        symbol = re.compile(r'\.|,|\(|\)|\"|\?|\!|\'s*|&|;|:') # 記号除去
        stop_words = "a,able,about,across,after,all,almost,also,am,among,an,and,any,are,as,at,be,because,been,but,by,can,cannot,could,dear,did,do,does,either,else,ever,every,for,from,get,got,had,has,have,he,her,hers,him,his,how,however,i,if,in,into,is,it,its,just,least,let,like,likely,may,me,might,most,must,my,neither,no,nor,not,of,off,often,on,only,or,other,our,own,rather,said,say,says,she,should,since,so,some,than,that,the,their,them,then,there,these,they,this,tis,to,too,twas,up,us,wants,was,we,were,what,when,where,which,while,who,whom,why,will,with,would,yet,you,your,one,--,movie,movies,film".split(',') # ストップワード除去 http://xpo6.com/list-of-english-stop-words/より
        lemmatizer = WordNetLemmatizer()
        
        voc = []
        with open(file_name, 'r')as txt:
            for t in txt:
                words = re.sub(symbol, '', t).strip().split(' ') # 記号除去
                words = list(filter(lambda x:len(x) > 1, words)) # 1文字を除去
                words = list(filter(lambda x:x not in stop_words, words)) # ストップワード除去
                words = [lemmatizer.lemmatize(w) for w in words] # 見出し語化
                self.corpus.append(words)
                voc += list(filter(lambda x:x not in voc, set(words)))
        self.voc_dic = dict(zip(voc, range(len(voc))))

    # 学習
    def train(self, iteration=10000):
        for ite in range(iteration):
            for i, m in enumerate(self.corpus):
                for j, n in enumerate(m):
                    z = self.topic[i][j]
                    self.increment(i, n, z, -1)
                    
                    # p(z|w)の計算 単語が分かっているときトピックzとなる確率
                    pz = self.n_wz[:,self.voc_dic[n]] * self.n_dz[i,:] / self.n_z / self.n_d[i]
                    # トピックを更新 トピックはランダムに選ばれるが選ばれやすさの割合は確率に基づく
                    z = np.random.multinomial(1, pz/pz.sum()).argmax()
                    
                    self.topic[i][j] = z
                    self.increment(i, n, z, 1)
                    
            if ite % 1000 == 0:
                print('iteration:{}'.format(ite))
                
    def increment(self, i, n, z, num):
        self.n_dz[i, z] += num
        self.n_wz[z, self.voc_dic[n]] += num
        self.n_d[i] += num
        self.n_z[z] += num

    # 各トピックの上位n単語を表示する
    def topword(self, num):
        for z in range(self.Z):
            word_rank = {}
            for w, i in self.voc_dic.items():
                word_rank[w] = float(self.n_wz[z, i]) / float(self.n_z[z])
            print('-----topic:{}-----'.format(z))
            cnt = 0
            for k, v in sorted(word_rank.items(), key=lambda x:x[1], reverse=True):
                print(k + ':' + str(v))
                cnt += 1
                if cnt == num:
                    break
        
def main():
    # movie2.txtファイルがBINARYであるので $nkf -w movie2.txt > movie2.txt.utf-8 でuft-8に変換したファイルを使用している
    file_train = 'movie2.txt.utf-8'

    lda = LDA(5)
    lda.set_initialize(file_train)
    lda.train(1)
    lda.topword(10)

if __name__ == '__main__':
    main()
