from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from itertools import combinations_with_replacement

import pandas as pd
import numpy as np

pd.set_option('display.width', 250)
pd.set_option('display.max_columns', 10)
pd.set_option('display.max_colwidth', 100)


LOC = './data/chally.csv'



class Team:
    def __init__(self, name):
        self.name = name
        self.games = []
        self.wins = 0
        self.losses = 0

    def _update_team(self):
        self.wins = sum([g.wins for g in self.games])
        self.losses = sum([4-g.wins for g in self.games])

class Game:
    def __init__(self, data):
        self.match_day = data['Match Day']
        self.date = data['Date']
        self.week_day = data['Weekday']
        self.home_name = data['Home']
        self.away_name = data['Away']
        self.home = teams[data['Home']]
        self.away = teams[data['Away']]
        self.result = data['True Result']
        self._update_teams()

    def _update_teams(self):
        home_wins = int(self.result.split('-')[0])
        away_wins = int(self.result.split('-')[1])
        home_game = TeamGame(wins=home_wins, vs=self.away, 
                match_day=self.match_day, date=self.date)
        away_game = TeamGame(wins=away_wins, vs=self.home, 
                match_day=self.match_day, date=self.date)
        self.home.games.append(home_game)
        self.away.games.append(away_game)
        self.home._update_team()
        self.away._update_team()


class TeamGame:
    def __init__(self, **kwargs):
        self.wins = kwargs['wins']
        self.vs = kwargs['vs']
        self.match_day = kwargs['match_day']
        self.date = kwargs['date']


def create_teams(df):
    """Create the team objects.

    Args:
        df: pandas DataFrame from which team data is derived.
    
    Returns:
        Dict of team objects for which the team names are the keys.
    """
    global teams

    team_names = sorted(list(set(list(df['Home'].values))))
    teams = {t: Team(t) for t in team_names} 
    parse_games(df)
    return teams


def parse_games(df):
    games = []
    cols = df.columns
    for row in df.values:
        data = {cols[i]: row[i] for i in range(len(cols))}
        games.append(Game(data))
    return games


def create_X_data():
    data = []
    for team_name in teams:
        team = teams[team_name]
        for g in team.games:
            row = [team.name, g.vs.name, g.wins]
            data.append(row)

    x_df = pd.DataFrame(data)
    x_df.columns = ['Team', 'VS', 'Result']
    return pd.get_dummies(x_df) 


def create_prediction_sheet(data_df):
    #model = LinearRegression()
    model = MLPRegressor()
    X_df = data_df[data_df.columns[1:]]
    Y_df = data_df['Result']

    model.fit(X_df.values, Y_df.values)

    team_names = sorted(list(set(list([t for t in teams]))))
    zeroes = [0 for i in range(len(team_names))]
    label_data = []
    for z in range(len(zeroes)):
        temp = zeroes.copy()
        temp[z] = 1
        label_data.append(temp)
    
    data = []
    for i in label_data:
        for j in label_data:
            team_name = team_names[np.argmax(i)]
            opponent_name = team_names[np.argmax(j)]
            pred_data = np.array(i + j).reshape(1, -1)
            prediction = model.predict(pred_data)[0]
            row = [team_name, opponent_name, prediction]
            data.append(row)
    df = pd.DataFrame(data)
    df.columns = ['Team', 'Opponent', 'Prediction']
    return df


def main():
    df = pd.read_csv(LOC)
    #df = df[df['Match Day'] > 6]
    teams = create_teams(df)
    data_df = create_X_data()
    preds = create_prediction_sheet(data_df)
    grp = preds.groupby('Team').mean().sort_values('Prediction', ascending=False)

    m = grp['Prediction']
    print(grp)


if __name__ == "__main__":
    main()
