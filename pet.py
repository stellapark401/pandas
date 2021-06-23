import numpy as np
import pandas as pd
from icecream import ic


class Quiz:
    q4_df: pd.DataFrame({})
    q28_df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    q30_df = pd.DataFrame({"customer_id": ['kim', 'lee', 'park', 'song', 'yoon', 'kang', 'tak', 'ryu', 'jang'],
                           "product_code": ['com', 'phone', 'tv', 'com', 'phone', 'tv', 'com', 'phone', 'tv'],
                           "grade": ['A', 'A', 'A', 'A', 'A', 'A', 'B', 'B', 'B'],
                           "purchase_amount": [30, 10, 0, 40, 15, 30, 0, 0, 10]})

    @staticmethod
    def quiz_2():
        ic(pd.__version__)

    @staticmethod
    def quiz_3():
        ic(pd.show_versions())

    def quiz_4(self):
        data = {'animal': ['cat', 'cat', 'snake', 'dog', 'dog', 'cat', 'snake', 'cat', 'dog', 'dog'],
                'age': [2.5, 3, 0.5, np.nan, 5, 2, 4.5, np.nan, 7, 3],
                'visits': [1, 3, 2, 3, 2, 3, 1, 1, 2, 1],
                'priority': ['yes', 'yes', 'no', 'yes', 'no', 'no', 'no', 'yes', 'no', 'no']}
        labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
        self.q4_df = pd.DataFrame(data, index=labels)
        if len(self.q4_df.index):
            print('a DataFrame has been created, go on.')

    def quiz_5(self):
        ic(self.q4_df.describe)

    def quiz_6(self):
        ic(self.q4_df.iloc[:3])

    def quiz_7(self):
        ic(self.q4_df.loc[:, ['animal', 'age']])

    def quiz_8(self):
        ic(self.q4_df.loc[[self.q4_df.index[3], self.q4_df.index[4], self.q4_df.index[8]], ['animal', 'age']])

    def quiz_9(self):
        df = self.q4_df.loc[self.q4_df['visits'] > 3]
        if len(df.index):
            ic(df)
        else:
            print('No animal has visited more than 3 times.')

    def quiz_10(self):
        ic(self.q4_df.loc[pd.isna(self.q4_df['age'])])

    def quiz_11(self):
        ic(self.q4_df.loc[self.q4_df['age'] < 3])

    def quiz_12(self):
        ic(self.q4_df.loc[(2 <= self.q4_df['age']) & (self.q4_df['age'] <= 4)])

    def quiz_13(self):
        self.q4_df.loc[['f'], ['age']] = 1.5
        ic(self.q4_df.loc['f'])

    def quiz_14(self):
        ic(self.q4_df['visits'].sum())

    def quiz_15(self):
        temp = self.q4_df['animal'].unique()
        ic([{temp[i]: self.q4_df.loc[self.q4_df['animal'] == temp[i],
                                     'age'].sum() / len(self.q4_df.loc[self.q4_df['animal'] == temp[i]])}
            for i, j in enumerate(temp)])

    def quiz_16(self):
        df = pd.concat([self.q4_df, pd.DataFrame({'animal': 'dog', 'age': 5.5, 'visits': 2, 'priority': 'no'},
                                                 index=['k'])])
        ic(df)
        ic(df.drop('k'))

    def quiz_17(self):
        ic(len(self.q4_df['animal'].unique()))

    def quiz_18(self):
        ic(self.q4_df.sort_values('age', ascending=False).sort_values('visits'))

    def quiz_19(self):
        df = self.q4_df
        df['priority'] = df['priority'] == 'yes'
        ic(df)

    def quiz_20(self):
        df = self.q4_df
        for i, j in enumerate(df.index):
            if df.iloc[i, 0] == 'snake':
                df.iloc[i, 0] = 'python'
        ic(df)

    # it didn't check to see if the given age is NaN, so the frequency is incorrect.
    def quiz_21_wo_pivot(self):
        df = self.q4_df
        # this can be replaced as df['animal'].replace('snake','python')
        for i, j in enumerate(df.index):
            if df.iloc[i, 0] == 'snake':
                df.iloc[i, 0] = 'python'
        df_cat = df.loc[df['animal'] == 'cat']
        df_python = df.loc[df['animal'] == 'python']
        df_dog = df.loc[df['animal'] == 'dog']
        temp_cat = df_cat['visits'].unique()
        temp_python = df_python['visits'].unique()
        temp_dog = df_dog['visits'].unique()
        cat_avg = [{f'visits: {temp_cat[i]}': df.loc[df_cat['visits'] == j, 'age'].sum() / len(
            df.loc[df_cat['visits'] == temp_cat[i]])} for i, j in enumerate(temp_cat)]
        python_avg = [{f'visits: {temp_python[i]}': df_python.loc[df['visits'] == j, 'age'].sum() / len(
            df_python.loc[df['visits'] == temp_python[i]])} for i, j in enumerate(temp_python)]
        dog_avg = [{f'visits: {temp_dog[i]}': df_dog.loc[df['visits'] == j, 'age'].sum() / len(
            df_dog.loc[df['visits'] == temp_dog[i]])} for i, j in enumerate(temp_dog)]
        ic(cat_avg)
        ic(python_avg)
        ic(dog_avg)

    def quiz_21_wh_pivot(self):
        df = self.q4_df
        avg_age = df.pivot_table(index='animal', columns='visits', values='age', aggfunc='mean')
        ic(avg_age)

    @staticmethod
    def quiz_22():
        df = pd.DataFrame({'A': [1, 2, 2, 3, 4, 5, 5, 5, 6, 7, 7]})
        ic(df)
        ic(df['A'].unique())

    @staticmethod
    def quiz_23():
        df = pd.DataFrame(np.random.random(size=(5, 3)))
        ic(df)
        for i in range(len(df.index)):
            for j in range(len(df.columns)):
                df.iloc[i, j] = df.iloc[i, j] - df.iloc[i].sum() / 3
        ic(df)

    @staticmethod
    def quiz_24_wo_idxmax():
        df = pd.DataFrame(np.random.random(size=(5, 10)), columns=list('abcdefghij'))
        ic(df)
        df_wh_sum = pd.concat([df, pd.DataFrame(pd.Series(df.sum()), columns=['sum']).transpose()])
        ic(df_wh_sum.transpose().sort_values('sum'))
        ic(df_wh_sum.transpose().sort_values('sum').iloc[0, 5])

    # 가장 작은 합을 가진 열을 출력하는 method (최대는 max)
    # df.sum().idxmax()
    @staticmethod
    def quiz_24_wh_idxmax():
        pass

    @staticmethod
    def quiz_25():
        df = pd.DataFrame(np.random.randint(0, 2, size=(10, 3)))
        ic(df)
        ic(len(df.drop_duplicates().index))

    @staticmethod
    def quiz_26():
        nan = np.nan
        data = [[0.04, nan, nan, 0.25, nan, 0.43, 0.71, 0.51, nan, nan],
                [nan, nan, nan, 0.04, 0.76, nan, nan, 0.67, 0.76, 0.16],
                [nan, nan, 0.5, nan, 0.31, 0.4, nan, nan, 0.24, 0.01],
                [0.49, nan, nan, 0.62, 0.73, 0.26, 0.85, nan, nan, nan],
                [nan, nan, 0.41, nan, 0.05, nan, 0.61, nan, 0.48, 0.68]]
        columns = list('abcdefghij')
        df = pd.DataFrame(data, columns=columns)
        where_is_nan = []
        for i in range(len(df.index)):
            cnt = 0
            for j, k in enumerate(list('abcdefghij')):
                if np.isnan(df.iloc[i, j]):
                    cnt += 1
                if cnt == 3:
                    where_is_nan.append(k)
                    break
        ic(where_is_nan)

    @staticmethod
    def quiz_27_wo_pivot():
        df = pd.DataFrame({'group': list('aaabbcaabcccbbc'),
                           'values': [12, 345, 3, 1, 45, 14, 4, 52, 54, 23, 235, 21, 57, 3, 87]})
        ic(df)
        max_value = []
        for i in 'a', 'b', 'c':
            max_value.append({i: df[df['group'] == i]['values'].max()})
        ic(max_value)

    @staticmethod
    def quiz_27_wh_pivot():
        df = pd.DataFrame({'group': list('aaabbcaabcccbbc'),
                           'values': [12, 345, 3, 1, 45, 14, 4, 52, 54, 23, 235, 21, 57, 3, 87]})
        ic(df.pivot(columns='group', values='values').idxmax())

    def quiz_28_wo_ndarray_method(self):
        df = self.q28_df
        conv = []
        for i, j in enumerate(df.columns):
            conv.append({df.columns[i]: df[j].values})
        ic(df)
        ic(conv)

    def quiz_28_wh_ndarray_method(self):
        df = self.q28_df
        # df.values is ndarray of numpy
        ic(df)
        ic(df.values)
        ic(df.values.tolist())

    def quiz_29_wo_df_method(self):
        df = self.q28_df
        conv = []
        for i, j in enumerate(df.columns):
            conv.append({df.columns[i]: df[j].values})
        ic(df)

    def quiz_29_wh_df_method(self):
        df = self.q28_df
        ic(df.to_dict())

    def quiz_30(self):
        df = self.q30_df
        ic(df)
        ic(df.pivot(index='customer_id', columns='product_code', values='purchase_amount'))

    def quiz_31(self):
        df = self.q30_df
        ic(df.pivot(columns='product_code', index=['customer_id', 'grade'], values='purchase_amount'))
        ic(df.pivot(columns='product_code', index=['grade', 'customer_id'], values='purchase_amount'))

    @staticmethod
    def main():
        qz = Quiz()
        while True:
            menu = int(input(' - pandas version\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t2\n'
                             ' - pandas libraries version\t\t\t\t\t\t\t\t\t\t\t\t\t3\n'
                             ' - create DataFrame\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t4\n'
                             ' - show the DataFrame\t\t\t\t\t\t\t\t\t\t\t\t\t\t5\n'
                             ' - show 3 of head\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t6\n'
                             ' - show animal&age col.\t\t\t\t\t\t\t\t\t\t\t\t\t\t7\n'
                             " - show 3rd, 4th, 8th rows' animal&age col.\t\t\t\t\t\t\t\t\t8\n"
                             ' - show rows where visits is more than 3\t\t\t\t\t\t\t\t\t9\n'
                             ' - show rows where age is NaN\t\t\t\t\t\t\t\t\t\t\t\t10\n'
                             ' - show rows where age is less than 3\t\t\t\t\t\t\t\t\t\t11\n'
                             ' - show rows where age is between 2 and 4\t\t\t\t\t\t\t\t\t12\n'
                             " - change f row's age to 1.5\t\t\t\t\t\t\t\t\t\t\t\t13\n"
                             " - show the sum of visits \t\t\t\t\t\t\t\t\t\t\t\t\t14\n"
                             " - show the average age of an animal\t\t\t\t\t\t\t\t\t\t15\n"
                             " - create 'k' row with 'dog', 'age=5.5',\n\t 'no priority', 'visits=1'"
                             "\t\t\t\t\t\t\t\t\t\t\t\t16\n\t  then delete the 'k' row back\n"
                             " - show the number of animal kinds in the table\t\t\t\t\t\t\t\t17\n"
                             " - align the rows age descending, visits ascending\t\t\t\t\t\t\t18\n"
                             " - map the priority col. as True for 'yes', False for 'no'\t\t\t\t\t19\n"
                             " - change 'snake' to 'python' in 'animal' col.\t\t\t\t\t\t\t\t20\n"
                             " - find average age for each animal and visit\t\t\t\t\t\t\t\t21\n"
                             " _ print col. 'A' with unique values from DataFrame\t\t\t\t\t\t\t22\n"
                             " - print row's value subtracted with row's average\t\t\t\t\t\t\t23\n"
                             " - select the row which has the smallest sum of values and print the sum\t24\n"
                             " - print the number of unique rows aka the rank of matrix\t\t\t\t\t25\n"
                             " - find which col. of a given row has third NaN, then print col. label\t\t26\n"
                             " - find the maximum value for each group\t\t\t\t\t\t\t\t\t27\n"
                             " - convert a DataFrame to List\t\t\t\t\t\t\t\t\t\t\t\t28\n"
                             " - convert a DataFrame to Dictionary\t\t\t\t\t\t\t\t\t\t29\n"
                             " - reshape the given table with customer_id as index, product_cod column,\n"
                             "\t  purchase_amount values\t\t\t\t\t\t\t\t\t\t\t\t30\n"
                             " - reshape the given table with customer_id and grade as index,\n"
                             "\t  product_cod column, purchase_amount values\t\t\t\t\t\t\t31\n"
                             '0. exit\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t0'))
            if menu == 0:
                break
            elif menu == 2:
                qz.quiz_2()
            elif menu == 3:
                qz.quiz_3()
            elif menu == 4:
                qz.quiz_4()
            elif menu == 5:
                qz.quiz_5()
            elif menu == 6:
                qz.quiz_6()
            elif menu == 7:
                qz.quiz_7()
            elif menu == 8:
                qz.quiz_8()
            elif menu == 9:
                qz.quiz_9()
            elif menu == 10:
                qz.quiz_10()
            elif menu == 11:
                qz.quiz_11()
            elif menu == 12:
                qz.quiz_12()
            elif menu == 13:
                qz.quiz_13()
            elif menu == 14:
                qz.quiz_14()
            elif menu == 15:
                qz.quiz_15()
            elif menu == 16:
                qz.quiz_16()
            elif menu == 17:
                qz.quiz_17()
            elif menu == 18:
                qz.quiz_18()
            elif menu == 19:
                qz.quiz_19()
            elif menu == 20:
                qz.quiz_20()
            elif menu == 21:
                qz.quiz_21_wh_pivot()
            elif menu == 22:
                qz.quiz_22()
            elif menu == 23:
                qz.quiz_23()
            elif menu == 24:
                qz.quiz_24_wo_idxmax()
                qz.quiz_24_wh_idxmax()
            elif menu == 25:
                qz.quiz_25()
            elif menu == 26:
                qz.quiz_26()
            elif menu == 27:
                qz.quiz_27_wo_pivot()
                qz.quiz_27_wh_pivot()
            elif menu == 28:
                qz.quiz_28_wo_ndarray_method()
                qz.quiz_28_wh_ndarray_method()
            elif menu == 29:
                qz.quiz_29_wo_df_method()
                qz.quiz_29_wh_df_method()
            elif menu == 30:
                qz.quiz_30()
            elif menu == 31:
                qz.quiz_31()


Quiz.main()
