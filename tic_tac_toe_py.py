import pandas as pd
import numpy as np
import random, pprint
from scipy.ndimage.interpolation import shift

from keras.models import Sequential,load_model
from keras.layers import Dense, Dropout
from keras.optimizers import SGD

#game class
class tictactoe(object):
    def __init__(self):
        #1:player1, 0:player2, 2:empty
        self.board= np.full((3,3),2)
    
    #who starts
    def whostart(self):
        turn = np.random.randint(0,2, size=1)
        if turn == 0:
            self.activeplayer = 0
        elif turn == 1:
            self.activeplayer = 1
        return self.activeplayer

    def move(self, player, coord):
        if self.board[coord] != self.gamestatus() != 'In Progress' or self.activeplayer != player:
            raise ValueError('Invalid Move')
        self.board[coord] = player
        self.activeplayer = 1 - player
        return self.gamestatus(), self.board

    def gamestatus(self):
        #win - vertical
        for i in range(self.board.shape[0]):
            if 2 not in self.board[i, :] and len(set(self.board[i, :])) == 1:
                return "win"
        #win - horizontal
        for  j in range(self.board.shape[1]):
            if 2 not in self.board[:, j] and len(set(self.board[:, j])) == 1:
                return "win"
        #win- cross
        if 2 not in np.diag(self.board) and len(set(np.diag(self.board))) == 1:
            return "win"
        
        if 2 not in np.diag(np.fliplr(self.board)) and len(set(np.diag(np.fliplr(self.board)))) == 1:
            return "win"

        #draw
        if 2 not in self.board:
            return "draw"
        
        #in progress
        else:
            return "In Progress"

def movegen(currentboardstate, activeplayer):
    avaliblemoves = {}
    for i in range( currentboardstate.shape[0]):
        for j in range(currentboardstate.shape[1]):
            if currentboardstate[i,j] == 2:
                #copied to avoid crushing object values
                boardstatecopy = currentboardstate.copy()
                boardstatecopy[i,j] = activeplayer
                avaliblemoves[(i,j)] = boardstatecopy.flatten()
    return avaliblemoves

#move selector
def movesel(model, currentboardstate, activeplayer):
    tracker = {}
    availablemoves = movegen(currentboardstate, activeplayer)
    for movecoord in availablemoves:
        score = model.predict(availablemoves[movecoord].reshape(1,9))
        tracker[movecoord] = score
    selectedmove = max(tracker, key=tracker.get)
    newboardstate = availablemoves[selectedmove]
    score = tracker[selectedmove]
    return selectedmove, newboardstate, score

#nn-model
model = Sequential()
model.add(Dense(18, input_dim=9, kernel_initializer='normal', activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(9, kernel_initializer='normal', activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(1, kernel_initializer='normal'))

#optimizer SGD
learningrate = 0.001
momentum=0.8
sgd =SGD(lr=learningrate, momentum=0.8, nesterov=False)

model.compile(loss='mean_squared_error', optimizer= sgd)
#model.summary()

#openent moves
#can we win after 1 move?
#vertical
def rowwincheck(currentboardstate, avaliblemoves, activeplayer):
    avaliblemovecoords= list(avaliblemoves.keys())
    random.shuffle(avaliblemovecoords)

    for coord in avaliblemovecoords:
        currentboardstatecopy = currentboardstate.copy()
        currentboardstatecopy[coord] = activeplayer
        for i in range(currentboardstatecopy.shape[0]):
            if 2 not in currentboardstatecopy[i, :] and len(set(currentboardstatecopy[i, :])) == 1:
                selectedmove = coord
                return selectedmove

#horizontal
def colwincheck(currentboardstate, avaliblemoves, activeplayer):
    avaliblemovecoords= list(avaliblemoves.keys())
    random.shuffle(avaliblemovecoords)

    for coord in avaliblemovecoords:
        currentboardstatecopy = currentboardstate.copy()
        currentboardstatecopy[coord] = activeplayer
        for i in range(currentboardstatecopy.shape[1]):
            if 2 not in currentboardstatecopy[:, i] and len(set(currentboardstatecopy[:, i])) == 1:
                selectedmove = coord
                return selectedmove

#diagonal
def diagwincheck(currentboardstate, avaliblemoves, activeplayer):
    avaliblemovecoords= list(avaliblemoves.keys())
    random.shuffle(avaliblemovecoords)

    for coord in avaliblemovecoords:
        currentboardstatecopy = currentboardstate.copy()
        currentboardstatecopy[coord] = activeplayer
        if 2 not in np.diag(currentboardstatecopy) and len(set(np.diag(currentboardstatecopy))) == 1:
            selectedmove = coord
            return selectedmove

#reverse diagonal
def diag2wincheck(currentboardstate, avaliblemoves, activeplayer):
    avaliblemovecoords= list(avaliblemoves.keys())
    random.shuffle(avaliblemovecoords)

    for coord in avaliblemovecoords:
        currentboardstatecopy = currentboardstate.copy()
        currentboardstatecopy[coord] = activeplayer
        if 2 not in np.diag(np.fliplr(currentboardstatecopy)) and len(set(np.diag(np.fliplr(currentboardstatecopy)))) == 1:
            selectedmove = coord
            return selectedmove

#if openent is winning block it
#vertical
def rowblockcheck(currentboardstate, avaliblemoves, activeplayer):
    avaliblemovecoords= list(avaliblemoves.keys())
    random.shuffle(avaliblemovecoords)

    for coord in avaliblemovecoords:
        currentboardstatecopy = currentboardstate.copy()
        currentboardstatecopy[coord] = activeplayer
        for i in range(currentboardstatecopy.shape[0]):
            if 2 not in currentboardstatecopy[i, :] and ((currentboardstatecopy[i, :] == 1).sum()) == 2:
                if not (2 not in currentboardstatecopy[i, :] and (currentboardstatecopy[i, :] == 1).sum() == 2):
                    selectedmove = coord
                    return selectedmove

#horizontal
def colblockcheck(currentboardstate, avaliblemoves, activeplayer):
    avaliblemovecoords= list(avaliblemoves.keys())
    random.shuffle(avaliblemovecoords)

    for coord in avaliblemovecoords:
        currentboardstatecopy = currentboardstate.copy()
        currentboardstatecopy[coord] = activeplayer
        for i in range(currentboardstatecopy.shape[0]):
            if 2 not in currentboardstatecopy[:, i] and ((currentboardstatecopy[:, i] == 1).sum()) == 2:
                if not (2 not in currentboardstatecopy[:, i] and (currentboardstatecopy[:, i] == 1).sum() == 2):
                    selectedmove = coord
                    return selectedmove

#diagonal
def diagblockcheck(currentboardstate, avaliblemoves, activeplayer):
    avaliblemovecoords= list(avaliblemoves.keys())
    random.shuffle(avaliblemovecoords)

    for coord in avaliblemovecoords:
        currentboardstatecopy = currentboardstate.copy()
        currentboardstatecopy[coord] = activeplayer
        if 2 not in np.diag(currentboardstatecopy) and ((np.diag(currentboardstatecopy) == 1).sum()) == 2:
            if not (2 not in np.diag(currentboardstatecopy) and (np.diag(currentboardstatecopy) == 1).sum() == 2):
                selectedmove = coord
                return selectedmove

#reverse diagonal
def diag2blockcheck(currentboardstate, avaliblemoves, activeplayer):
    avaliblemovecoords= list(avaliblemoves.keys())
    random.shuffle(avaliblemovecoords)

    for coord in avaliblemovecoords:
        currentboardstatecopy = currentboardstate.copy()
        currentboardstatecopy[coord] = activeplayer
        if 2 not in np.diag(np.fliplr(currentboardstatecopy)) and ((np.diag(np.fliplr(currentboardstatecopy)) == 1).sum()) == 2:
            if not (2 not in np.diag(np.fliplr(currentboardstatecopy)) and (np.diag(np.fliplr(currentboardstatecopy)) == 1).sum() == 2):
                selectedmove = coord
                return selectedmove

#can we win after 2 moves?
#vertical
def row2movecheck(currentboardstate, avaliblemoves, activeplayer):
    avaliblemovecoords= list(avaliblemoves.keys())
    random.shuffle(avaliblemovecoords)

    for coord in avaliblemovecoords:
        currentboardstatecopy = currentboardstate.copy()
        currentboardstatecopy[coord] = activeplayer
        for i in range(currentboardstatecopy.shape[0]):
            if 1 not in currentboardstatecopy[i, :] and ((currentboardstatecopy[i, :] == 0).sum()) == 2:
                if not (1 not in currentboardstatecopy[i, :] and (currentboardstatecopy[i, :] == 0).sum() == 2):
                    selectedmove = coord
                    return selectedmove

#horizontal
def col2movecheck(currentboardstate, avaliblemoves, activeplayer):
    avaliblemovecoords= list(avaliblemoves.keys())
    random.shuffle(avaliblemovecoords)

    for coord in avaliblemovecoords:
        currentboardstatecopy = currentboardstate.copy()
        currentboardstatecopy[coord] = activeplayer
        for i in range(currentboardstatecopy.shape[0]):
            if 1 not in currentboardstatecopy[:, i] and ((currentboardstatecopy[:, i] == 0).sum()) == 2:
                if not (1 not in currentboardstatecopy[:, i] and (currentboardstatecopy[:, i] == 0).sum() == 2):
                    selectedmove = coord
                    return selectedmove

#diagonal
def diag2movecheck(currentboardstate, avaliblemoves, activeplayer):
    avaliblemovecoords= list(avaliblemoves.keys())
    random.shuffle(avaliblemovecoords)

    for coord in avaliblemovecoords:
        currentboardstatecopy = currentboardstate.copy()
        currentboardstatecopy[coord] = activeplayer
        if 1 not in np.diag(currentboardstatecopy) and ((np.diag(currentboardstatecopy) == 0).sum()) == 2:
            if not (1 not in np.diag(currentboardstatecopy) and (np.diag(currentboardstatecopy) == 0).sum() == 2):
                selectedmove = coord
                return selectedmove

#reverse diagonal
def diag22movecheck(currentboardstate, avaliblemoves, activeplayer):
    avaliblemovecoords= list(avaliblemoves.keys())
    random.shuffle(avaliblemovecoords)

    for coord in avaliblemovecoords:
        currentboardstatecopy = currentboardstate.copy()
        currentboardstatecopy[coord] = activeplayer
        if 1 not in np.diag(np.fliplr(currentboardstatecopy)) and ((np.diag(np.fliplr(currentboardstatecopy)) == 0).sum()) == 2:
            if not (1 not in np.diag(np.fliplr(currentboardstatecopy)) and (np.diag(np.fliplr(currentboardstatecopy)) == 0).sum() == 2):
                selectedmove = coord
                return selectedmove

#openent move selector
def openentmovesel(currentboardstate, activeplayer, mode):
    availablemoves = movegen(currentboardstate, activeplayer)

    winmovechecks = [rowwincheck, colwincheck, diagwincheck, diag2wincheck]
    blockmovechecks = [rowblockcheck, colblockcheck, diagblockcheck, diag2blockcheck]
    secondmoveschecks = [row2movecheck, col2movecheck, diag2movecheck, diag22movecheck]
    
    if mode== 'random':
        
        selectmove= random.choice(list(availablemoves.keys()))
        return selectmove
    elif mode== 'hard':
        random.shuffle(winmovechecks)
        random.shuffle(blockmovechecks)
        random.shuffle(secondmoveschecks)

        for fn in winmovechecks:
            if fn(currentboardstate, availablemoves, activeplayer):
                return fn (currentboardstate,availablemoves,activeplayer)
        
        for fn in blockmovechecks:
            if fn(currentboardstate, availablemoves, activeplayer):
                return fn (currentboardstate,availablemoves,activeplayer)
        
        for fn in secondmoveschecks:
            if fn(currentboardstate, availablemoves, activeplayer):
                return fn (currentboardstate,availablemoves,activeplayer)

        selectedmove = random.choice(list(availablemoves.keys()))
        return selectedmove

# AI Train
#------------------

def train(model, mode, print_progress= False):
    if print_progress == True:
        print('progress started')
    game = tictactoe()
    game.whostart()
    scorelist= []
    correctedscorelist= []
    newboardstatelist= []

    while(1):
        if game.gamestatus() == 'In Progress' and game.activeplayer == 1:
            #move for ai, use move selector
            selectedmove, newboardstate, score = movesel(model, game.board, game.activeplayer)
            scorelist.append(score[0][0])
            newboardstatelist.append(newboardstate)
            # next move
            gamestatus, board = game.move(game.activeplayer, selectedmove)
            if print_progress == True:
                print('ai move')
                print(board, '\n')
        elif game.gamestatus() == 'In Progress' and game.activeplayer == 0:
                selectedmove = openentmovesel(game.board, game.activeplayer, mode= mode)
                #next move
                gamestatus, board = game.move(game.activeplayer, selectedmove)
                if print_progress == True:
                    print('oponents move')
                    print(board, '\n')
        else:
            break

    # scores          1 / 0 / -1 
    newboardstatelist = tuple(newboardstatelist)
    newboardstatelist = np.vstack(newboardstatelist)
    if gamestatus == 'win' and (1-game.activeplayer) == 1:
        correctedscorelist = shift(scorelist, -1, cval= 1.0)
        result = 'win'
    if gamestatus == 'win' and (1-game.activeplayer) != 1:
        correctedscorelist = shift(scorelist, -1, cval= -1.0)
        result = 'lost'
    if gamestatus == 'draw':
        correctedscorelist = shift(scorelist, -1, cval = 0.0)
        result = 'draw'
    if print_progress == True:
        print('AI:', result)
        print('-----------------')

    x = newboardstatelist
    y = correctedscorelist

    def unisonshuffledcopies(a,b):
        assert len(a) == len(b)
        p = np.random.permutation(len(a))
        return a[p], b[p]
    
    x,y = unisonshuffledcopies(x,y)
    x = x.reshape(-1, 9)
    # model fitting
    model.fit(x, y, epochs= 1, batch_size=1, verbose=0)
    return model, y, result 

#train for one time and update
#updatemodel, y, result = train(model, mode='hard', print_progress=True)

#train with many games
gamecounter = 1
modelist = ['random', 'hard']
while(gamecounter<1000000):
    modeselected = np.random.choice(modelist, 1, p=[0.5, 0.5])
    model, y, result = train(model, mode=modeselected[0], print_progress=False)
    if gamecounter % 5 ==0:
        print('game # {}', format(gamecounter))
        print('mode: {}', format(modeselected[0]))
        print('result # {}', format(result))
    gamecounter += 1

#save model
model.save('tictactoe_model_1000iteration.h5')

#use trained model
#model = load_model('tictactoe_model.h5')


def gametest():
    #obj
    game = tictactoe()
    game.whostart()
    print('player:', game.activeplayer)
    print('board initialize\n', game.board)

    gamestatus, board= game.move(game.activeplayer, (0,0))
    print('new board status\n', game.board)
    print(gamestatus)
    print('posible moves:',movegen(game.board, game.activeplayer)) 

    gamestatus, board= game.move(game.activeplayer, (0,1))
    print('new board status\n', game.board)
    print(gamestatus)
    print('posible moves:',movegen(game.board, game.activeplayer)) 

    gamestatus, board= game.move(game.activeplayer, (1,1))
    print('new board status\n', game.board)
    print(gamestatus)
    print('posible moves:',movegen(game.board, game.activeplayer)) 

#gametest()

def moveselectortest():
    game=tictactoe()
    game.whostart()
    print('player:', game.activeplayer)
    print('board:\n', game.board)
    selectedmove, newboardstate, score = movesel(model, game.board, game.activeplayer)
    print('move:', selectedmove)
    print(newboardstate.reshape(3,3))
    print('score:', score)

#moveselectortest()
