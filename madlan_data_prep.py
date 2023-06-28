import numpy as np
import pandas as pd
import re
from datetime import datetime


def prepare_data(data):
    data.loc[:, 'price'] = pd.to_numeric(data['price'], errors='coerce')
    data.dropna(subset=['price'], inplace=True)
    data.loc[:, 'Area'] = pd.to_numeric(data['Area'], errors='coerce')

    data['price'] = data['price'].replace('[^0-9]', '', regex=True)
    data['type'] = data['type'].replace('[^\w\s]', '', regex=True)
    data['Street'] = data['Street'].replace('[^\w\s]', '', regex=True)
    data['city_area'] = data['city_area'].replace('[^\w\s]', '', regex=True)
    data['description '] = data['description '].replace('[^\w\s]', '', regex=True)
    data['Area'] = data['Area'].replace('[^0-9]', '', regex=True)
    data['room_number'] = data['room_number'].replace('[^0-9]', '', regex=True)

    data['floor'] = data['floor_out_of'].str.split().str[1]
    data['total_floors'] = data['floor_out_of'].str.split().str[3]

    def categorize_entrance_date(date):
        if date == 'גמיש' or date == 'גמיש ':
            return 'flexible'
        elif date == 'מיידי':
            return 'less_than_6_months'
        elif date == 'לא צויין':
            return 'not_defined'
        elif isinstance(date, datetime):
            current_date = datetime.now().date()
            difference = (date.date() - current_date).days
            if difference < 0:
                return 'No'
            elif difference < 180:
                return 'less_than_6_months'
            elif difference < 365:
                return 'months_6_12'
            else:
                return 'above_year'
        else:
            try:
                parsed_date = datetime.strptime(date, '%Y-%m-%d')
                current_date = datetime.now().date()
                difference = (parsed_date.date() - current_date).days
                if difference < 0:
                    return 'No'
                elif difference < 180:
                    return 'less_than_6_months'
                elif difference < 365:
                    return 'months_6_12'
                else:
                    return 'above_year'
            except ValueError:
                return date

    data['entranceDate '] = data['entranceDate '].apply(categorize_entrance_date)

    data['handicapFriendly '] = data['handicapFriendly '].map(
        {'לא': 0, 'לא נגיש': 0, 'לא נגיש לנכים': 0, False: 0, 'נגיש': 1, 'נגיש לנכים': 1, True: 1})
    data['handicapFriendly '] = data['handicapFriendly '].fillna(0).astype(int)

    data['hasMamad '] = data['hasMamad '].replace(['no', 'אין', 'אין ממ"ד', 'לא', 'אין ממ״ד', False], 0)
    data['hasMamad '] = data['hasMamad '].replace(['יש ממ"ד', 'יש', 'יש ממ״ד', 'yes', 'כן', True], 1)
    data['hasMamad '] = data['hasMamad '].fillna(0).astype(int)

    data['hasBalcony '] = data['hasBalcony '].replace(['no', 'אין', 'לא', 'אין מרפסת', False], 0)
    data['hasBalcony '] = data['hasBalcony '].replace(['יש מרפסת', 'יש', 'yes', 'כן', True], 1)
    data['hasBalcony '] = data['hasBalcony '].fillna(0).astype(int)

    data['hasAirCondition '] = data['hasAirCondition '].replace(['no', 'אין', 'לא', 'אין מיזוג אויר', False], 0)
    data['hasAirCondition '] = data['hasAirCondition '].replace(
        ['יש מיזוג אוויר', 'יש מיזוג אויר', 'יש', 'yes', 'כן', True], 1)
    data['hasAirCondition '] = data['hasAirCondition '].fillna(0).astype(int)

    data['hasStorage '] = data['hasStorage '].replace(['no', 'אין', 'לא', 'אין מחסן', False], 0)
    data['hasStorage '] = data['hasStorage '].replace(['יש מחסן', 'יש', 'yes', 'כן', True], 1)
    data['hasStorage '] = data['hasStorage '].fillna(0).astype(int)

    data['hasBars '] = data['hasBars '].replace(['no', 'אין', 'לא', 'אין סורגים', False], 0)
    data['hasBars '] = data['hasBars '].replace(['יש סורגים', 'יש', 'yes', 'כן', True], 1)
    data['hasBars '] = data['hasBars '].fillna(0).astype(int)

    data['hasParking '] = data['hasParking '].replace(['no', 'אין', 'לא', 'אין חניה', False], 0)
    data['hasParking '] = data['hasParking '].replace(['יש חנייה', 'יש', 'yes', 'יש חניה', 'כן', True], 1)
    data['hasParking '] = data['hasParking '].fillna(0).astype(int)

    data['hasElevator '] = data['hasElevator '].replace(['no', 'אין', 'לא', 'אין מעלית', False], 0)
    data['hasElevator '] = data['hasElevator '].replace(['יש מעלית', 'יש', 'yes', 'כן', True], 1)
    data['hasElevator '] = data['hasElevator '].fillna(0).astype(int)

    data['type'] = data['type'].replace('בניין', 'דירה', regex=True)
    data['type'] = data['type'].replace('נחלה', 'מגרש', regex=True)

    data['City'] = data['City'].replace('נהריה', 'נהריה ', regex=True)
    data['City'] = data['City'].replace('נהרייה', 'נהריה', regex=True)

    data['city_area'] = data['city_area'].str.strip()

    data['condition '] = data['condition '].replace(False, 'לא צויין', regex=True)

    data.loc[:, 'room_number'] = pd.to_numeric(data['room_number'], errors='coerce')

    columns = ['City', 'price', 'condition ', 'room_number', 'Area', 'furniture ', 'total_floors', 'hasMamad ','hasElevator ']

    return data[columns]