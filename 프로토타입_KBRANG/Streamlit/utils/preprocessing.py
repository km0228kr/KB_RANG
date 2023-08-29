import pandas as pd
from sklearn.preprocessing import LabelEncoder

class UserLabelProcessor:
    def __init__(self, input_csv_path, user_csv_path, output_csv_path):
        self.input_csv_path = input_csv_path
        self.user_csv_path = user_csv_path
        self.output_csv_path = output_csv_path

    def melt_and_save(self):
        # Read user information from the CSV file
        df = pd.read_csv(self.input_csv_path)

        # Melt the DataFrame to create the rating DataFrame
        label_df = pd.melt(df, id_vars=["user_id"], value_vars=["금융", "증시", "부동산", "국제경제", "소비자", "경제/정책"],
                          var_name="label", value_name="rating")
        
        click_df = pd.melt(df, id_vars=["user_id"], 
                           value_vars=["금융_click_probs", "증시_click_probs", "부동산_click_probs", "국제경제_click_probs",
                                       "소비자_click_probs", "경제/정책_click_probs"],
                          var_name="label", value_name="click_probs")
        
        # read user information
        user = pd.read_csv(self.user_csv_path)
        # drop columns
        user = user.drop(["금융", "증시", "부동산", "국제경제", "소비자", "경제/정책", 
                          "금융_click_probs", "증시_click_probs", "부동산_click_probs", 
                          "국제경제_click_probs", "소비자_click_probs", "경제/정책_click_probs"], axis=1)
        
        # label encoding for categorical columns
        cat_col = ["gender", "age", "occupation", "address"]        
        for cat in cat_col:
            encoder = LabelEncoder()
            user[cat] = encoder.fit_transform(user[cat])
        
        label_df = pd.merge(label_df, user)
        click_df = pd.merge(click_df, user)
        
        label_click_df = pd.concat([label_df, click_df[["click_probs"]]], axis=1)

        # Save the rating DataFrame to another CSV file
        label_click_df.to_csv(self.output_csv_path, index=False)