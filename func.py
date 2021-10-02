#!/usr/bin/env python
# coding: utf-8
# %%
import numpy as np
import pandas as pd
from scipy import sparse


# %%
def clear_df(df, suffixes=['_x', '_y'], inplace=True):
    '''
    clear_df(df, suffixes=['_x', '_y'], inplace=True)
        Удаляет из входного df все колонки, оканчивающиеся на заданные суффиксы. 
        
        Parameters
        ----------
        df : pandas.DataFrame
        
        suffixies : Iterable, default=['_x', '_y']
            Суффиксы колонок, подлежащих удалению
            
        inplace : bool, default=True
            Нужно ли удалить колонки "на месте" или же создать копию DataFrame.
            
        Returns
        -------
        pandas.DataFrame (optional)
            df с удалёнными колонками
    '''
    
    def bad_suffix(column):
        nonlocal suffixes
        return any(column.endswith(suffix) for suffix in suffixes)
        
    columns_to_drop = [col for col in df.columns if bad_suffix(col)]
    return df.drop(columns_to_drop, axis=1, inplace=inplace)


def extract_unique(reviews, column): 
    '''
    extract_unique(reviews, column)
        Извлекает уникальные значения из колонки в DataFrame.
        
        Parameters
        ----------
        reviews : pandas.DataFrame
            pandas.DataFrame, из которого будут извлечены значения.
        
        column : str
            Имя колонки в <reviews>.
        
        Returns
        -------
        pandas.DataFrame
            Содержит одну именованную колонку с уникальными значениями. 
    '''
    
    unique = reviews[column].unique()
    return pd.DataFrame({column: unique})


def count_unique(reviews, column):
    '''
    count_unique(reviews, column)
        Извлекает и подсчитывает уникальные значения из колонки в DataFrame.
        
        Parameters
        ----------
        reviews : pandas.DataFrame
            pandas.DataFrame, из которого будут извлечены значения.
        
        column : str
            Имя колонки в <reviews>.
        
        Returns
        -------
        pandas.DataFrame
            Содержит две колонки: с уникальными значениями и счётчиком встреченных. 
    '''
    
    return reviews[column].value_counts().reset_index(name='count').rename({'index': column}, axis=1)



def filter_reviews(reviews, users=None, orgs=None): 
    '''
    filter_reviews(reviews, users=None, orgs=None)
    Оставляет в выборке только отзывы, оставленные заданными пользователями на заданные организации. 
    
    Parameters
    ----------
        users: pandas.DataFrame, default=None
            DataFrame, содержащий колонку <user_id>.
            Если None, то фильтрация не происходит. 
            
        orgs: pandas.DataFrame, default=None
            DataFrame, содержащий колонку <org_id>.
            Если None, то фильтрация не происходит. 
    
    Returns
    -------
        pandas.DataFrame
            Отфильтрованная выборка отзывов. 

    '''
    if users is not None: 
        reviews = reviews.merge(users, on='user_id', how='inner')
        clear_df(reviews)
        
    if orgs is not None:
        reviews = reviews.merge(orgs, on='org_id', how='inner')
        clear_df(reviews)
        
    return reviews


def train_test_split(reviews, ts_start, ts_end=None):
    '''
    train_test_split(reviews, ts_start, ts_end=None)
        Разделяет выборку отзывов на две части: обучающую и тестовую. 
        В тестовую выборку попадают только отзывы с user_id и org_id, встречающимися в обучающей выборке.

        Parameters
        ----------
        reviews : pandas.DataFrame 
            Отзывы из reviews.csv с обязательными полями:
                <rating>, <ts>, <user_id>, <user_city>, <org_id>, <org_city>.

        ts_start : int
            Первый день отзывов из тестовой выборки (включительно).

        ts_end : int, default=None
            Последний день отзывов из обучающей выборки (включительно)
            Если параметр равен None, то ts_end == reviews['ts'].max(). 

        Returns
        -------
        splitting : tuple
            Кортеж из двух pandas.DataFrame такой же структуры, как и reviews:
            в первом отзывы, попавшие в обучающую выборку, во втором - в тестовую.
    '''
    
    if not ts_end:
        ts_end = reviews['ts'].max()

    
    reviews_train = reviews[(reviews['ts'] < ts_start) | (reviews['ts'] > ts_end)]
    reviews_test = reviews[(ts_start <= reviews['ts']) & (reviews['ts'] <= ts_end)]
    
    # 1. Выбираем только отзывы на понравившиеся места у путешественников
    reviews_test = reviews_test[reviews_test['rating'] >= 4.0]
    reviews_test = reviews_test[reviews_test['user_city'] != reviews_test['org_city']]
    
    # 2. Оставляем в тесте только тех пользователей и организации, которые встречались в трейне
    train_orgs = extract_unique(reviews_train, 'org_id')
    train_users = extract_unique(reviews_train, 'user_id')
    
    reviews_test = filter_reviews(reviews_test, orgs=train_orgs)

    return reviews_train, reviews_test


def process_reviews(reviews):
    '''
    process_reviews(reviews)
        Извлекает из набора отзывов тестовых пользователей и таргет. 
        
        Parameters
        ----------
        reviews : pandas.DataFrame
            DataFrame с отзывами, содержащий колонки <user_id> и <org_id>
        
        Returns
        -------
        X : pandas.DataFrame
            DataFrame такой же структуры, как и в test_users.csv
            
        y : pandas.DataFrame
            DataFrame с колонками <user_id> и <target>. 
            В <target> содержится список org_id, посещённых пользователем. 
    '''
    
    
    y = reviews.groupby('user_id')['org_id'].apply(list).reset_index(name='target')
    X = pd.DataFrame(y['user_id'])
    
    return X, y


# %%
# ----------------- Метрика ---------------

def MNAP(size=20):
    '''
    MNAP(size=20)
        Создаёт метрику под <size> сделанных предсказаний.
        
        Parameters
        ----------
        size : int, default=20
            Размер рекомендованной выборки для каждого пользователя
        
        Returns
        -------
        func(pd.DataFrame, pd.DataFrame) -> float
            Функция, вычисляющая MNAP.
        
    '''
    
    assert size >= 1, "Size must be greater than zero!"
    
    def metric(y_true, predictions, size=size):
        '''
        metric(y_true, predictions, size=size)
            Метрика MNAP для двух перемешанных наборов <y_true> и <y_pred>.
            
            Parameters
            ----------
            y_true : pd.DataFrame
                DataFrame с колонками <user_id> и <target>. 
                В <target> содержится список настоящих org_id, посещённых пользователем. 
                
            predictions : pd.DataFrame
                DataFrame с колонками <user_id> и <target>. 
                В <target> содержится список рекомендованных для пользователя org_id.
                
            Returns
            -------
            float 
                Значение метрики.
        '''
        
        y_true = y_true.rename({'target': 'y_true'}, axis='columns')
        predictions = predictions.rename({'target': 'predictions'}, axis='columns')
        
        merged = y_true.merge(predictions, left_on='user_id', right_on='user_id')
    
        def score(x, size=size):
            '''
            Вспомогательная функция.
            '''
            
            
            y_true = x[1][1]
            predictions = x[1][2][:size]
            
            weight = 0
            
            inner_weights = [0]
            for n, item in enumerate(predictions):
                inner_weight = inner_weights[-1] + (1 if item in y_true else 0)
                inner_weights.append(inner_weight)
            
            for n, item in enumerate(predictions):                
                if item in y_true:
                    weight += inner_weights[n + 1] / (n + 1)
                    
            return weight / min(len(y_true), size)
    
        return np.mean([score(row) for row in merged.iterrows()])
    
        
    return metric


def print_score(score):
    print(f"Score: {score*100.0:.2f}")


# %%
def extract_top_by_rubrics(orgs, reviews, N):
    '''
    extract_top_by_rubrics(reviews, N)
        Набирает самые популярные организации по рубрикам, сохраняя распределение.
        
        Parameters
        ----------
        reviews : pd.DataFrame
            Отзывы пользователей для рекомендации.
            
        N : int
            Число рекомендаций.
        
        Returns
        -------
        orgs_list : list
            Список отобранных организаций.
    '''
    
    # извлечение популярных рубрик
    reviews = reviews.merge(orgs, on='org_id')[['org_id', 'rubrics_id']]
    
    rubrics = reviews.explode('rubrics_id').groupby('rubrics_id').size()
    rubrics = (rubrics / rubrics.sum() * N).apply(round).sort_values(ascending=False)

    # вывод списка рубрик по убыванию популярности
#     print(
#         pd.read_csv('data/rubrics.csv')
#         .merge(rubrics.reset_index(), left_index=True, right_on='rubrics_id')
#         .sort_values(by=0, ascending=False)[['rubric_id', 0]]
#     )
    
    # извлечение популярных организаций
    train_orgs = reviews.groupby('org_id').size().reset_index(name='count').merge(orgs, on='org_id')
    train_orgs = train_orgs[['org_id', 'count', 'rubrics_id']]

    most_popular_rubric = lambda rubrics_id: max(rubrics_id, key=lambda rubric_id: rubrics[rubric_id])
    train_orgs['rubrics_id'] = train_orgs['rubrics_id'].apply(most_popular_rubric)
    
    orgs_by_rubrics = train_orgs.sort_values(by='count', ascending=False).groupby('rubrics_id')['org_id'].apply(list)
    
    # соберём самые популярные организации в рубриках в один список
    
    orgs_list = []

    for rubric_id, count in zip(rubrics.index, rubrics):
        if rubric_id not in orgs_by_rubrics:
            continue 

        orgs_list.extend(orgs_by_rubrics[rubric_id][:count])
    
    return orgs_list


# %% [markdown]
# # memory-based

# %%
def reduce_reviews(reviews, min_user_reviews=5, min_org_reviews=13):
    '''
    reduce_reviews(reviews, min_user_reviews=5, min_org_reviews=13)
        Убирает из выборки пользователей и организации, у которых менее <min_reviews> отзывов в родном городе. 
        Оставляет только отзывы туристов. 
        
        Parameters
        ----------
        reviews : pandas.DataFrame 
            Выборка отзывов с обязательными полями:
                <user_id>, <user_city>.
        
        min_user_reviews : int, default=5
            Минимальное количество отзывов у пользователя, необходимое для включения в выборку.
            
        min_org_reviews : int, default=13
            Минимальное количество отзывов у организации, необходимое для включения в выборку.
            
        Returns
        -------
        splitting : tuple
            Кортеж из двух наборов.
            Каждый набор содержит 2 pandas.DataFrame:
                1. Урезанная выборка отзывов
                2. Набор уникальных организаций
                
            Первый набор содержит DataFrame-ы, относящиеся к отзывам, оставленным в родном городе, а второй -
            к отзывам, оставленным в чужом городе. ё
            
        users : pd.DataFrame
            Набор уникальных пользователей в выборке
        
    '''
    
    inner_reviews = reviews[reviews['user_city'] == reviews['org_city']]
    outer_reviews = reviews[reviews['user_city'] != reviews['org_city']]

    # оставляем только отзывы туристов на родной город 
    tourist_users = extract_unique(outer_reviews, 'user_id')
    inner_reviews = filter_reviews(inner_reviews, users=tourist_users)
    
    # выбираем только тех пользователей и организации, у которых есть <min_reviews> отзывов
    top_users = count_unique(inner_reviews, 'user_id')
    top_users = top_users[top_users['count'] >= min_user_reviews]
        
    top_orgs = count_unique(inner_reviews, 'org_id')
    top_orgs = top_orgs[top_orgs['count'] >= min_org_reviews]
        
    inner_reviews = filter_reviews(inner_reviews, users=top_users, orgs=top_orgs)
    outer_reviews = filter_reviews(outer_reviews, users=top_users)
    
    # combine reviews
    reviews = pd.concat([inner_reviews, outer_reviews])
    users = extract_unique(reviews, 'user_id')
    orgs = extract_unique(reviews, 'org_id')
    
    
    return (
        (
            inner_reviews,
            extract_unique(inner_reviews, 'org_id')
        ),
        (
            outer_reviews,
            extract_unique(outer_reviews, 'org_id')
        ),
        extract_unique(inner_reviews, 'user_id')
    )


# %%
def create_mappings(df, column):
    '''
    create_mappings(df, column)
        Создаёт маппинг между оригинальными ключами словаря и новыми порядковыми.
        
        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame с данными.
            
        column : str
            Название колонки, содержащей нужны ключи. 
        
        Returns
        -------
        code_to_idx : dict
            Словарь с маппингом: "оригинальный ключ" -> "новый ключ".
        
        idx_to_code : dict
            Словарь с маппингом: "новый ключ" -> "оригинальный ключ".
    '''
    
    code_to_idx = {}
    idx_to_code = {}
    
    for idx, code in enumerate(df[column].to_list()):
        code_to_idx[code] = idx
        idx_to_code[idx] = code
        
    return code_to_idx, idx_to_code


def map_ids(row, mapping):
    '''
    Вспомогательная функция
    '''
    
    return mapping[row]


def interaction_matrix(train_reviews,
                       reviews, test_users,
                       min_user_reviews = 5,
                       min_org_reviews = 12): 
    '''
    interaction_matrix(reviews, test_users, min_user_reviews=5, min_org_reviews=12)
        Создаёт блочную матрицу взаимодействий (вид матрицы описан в Returns)
        
        Parameters
        ----------
        reviews : pd.DataFrame
            Отзывы пользователей для матрицы взаимодействий.
            
        test_users : pd.DataFrame
            Пользователи, для которых будет выполнятся предсказание. 
        
        min_user_reviews : int, default=5
            Минимальное число отзывов от пользователя, необходимое для включения его в матрицу.
        
        min_org_reviews : int, default=12
            Минимальное число отзывов на организацию, необходимое для включения её в матрицу.
    
        Returns
        -------
        InteractionMatrix : scipy.sparse.csr_matrix
            Матрица, содержащая рейтинги, выставленные пользователями.
            Она блочная и имеет такой вид:
                 ---------------------------------------------------
                | TRAIN USERS, INNER ORGS | TRAIN USERS, OUTER ORGS |
                |                         |                         |
                 ---------------------------------------------------
                |  TEST USERS, INNER ORGS |  TEST USERS, OUTER ORGS |
                |                         |                         |
                 ---------------------------------------------------

        splitting : tuple
            Кортеж, содержащий два целых числа: 
                1. Число пользователей в обучающей выборке 
                2. Число организаций в домашнем регионе

        splitting: tuple
            Кортеж, содержащий два котрежа из двух словарей:
                1. (idx_to_uid, uid_to_idx) - содержит маппинг индекса к user_id
                2. (idx_to_oid, oid_to_idx) - содержит маппинг индекса к org_id
    '''
    
    
    info = reduce_reviews(train_reviews, min_user_reviews, min_org_reviews)
    (inner_reviews, inner_orgs), (outer_reviews, outer_orgs), train_users = info
    
    # удалим из обучающей выборки пользователей, которые есть в тестовой
    test_users = test_users[['user_id']]
    
    train_users = (
        pd.merge(train_users, test_users, indicator=True, how='outer')
        .query('_merge=="left_only"')
        .drop('_merge', axis=1)
    )
    
    inner_reviews = filter_reviews(inner_reviews, train_users)
    outer_reviews = filter_reviews(outer_reviews, train_users)
    
    # оставляем отзывы, оставленные тестовыми пользователями
    test_reviews = filter_reviews(reviews, test_users, pd.concat([inner_orgs, outer_orgs]))
    
    # получаем полный набор маппингов
    all_users = pd.concat([train_users, test_users])
    all_orgs = pd.concat([inner_orgs, outer_orgs])
    
    uid_to_idx, idx_to_uid = create_mappings(all_users, 'user_id')
    oid_to_idx, idx_to_oid = create_mappings(all_orgs, 'org_id')
    
    # собираем матрицу взаимодействий 
    reviews = pd.concat([inner_reviews, outer_reviews, test_reviews])    
        
    I = reviews['user_id'].apply(map_ids, args=[uid_to_idx]).values
    J = reviews['org_id'].apply(map_ids, args=[oid_to_idx]).values
    values = reviews['rating']
        
    interactions = sparse.coo_matrix(
        (values, (I, J)), 
        shape=(len(all_users), len(all_orgs)), 
        dtype=np.float64
    ).tocsr()
    
    
    return (
        interactions, 
        (len(train_users), len(inner_orgs)), 
        (
            (idx_to_uid, uid_to_idx),
            (idx_to_oid, oid_to_idx)
        )
    )

# %%

# %%

# %%
