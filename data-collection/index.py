from SmartApi import SmartConnect
import pyotp
from datetime import datetime, timedelta
import json
import time
import os

def append_json_to_file(file_path, new_data):
    existing_data = []
    
    try:
        # Check if file exists and read existing data
        if os.path.exists(file_path):
            with open(file_path, 'r') as file:
                content = file.read().strip()
                if content:
                    existing_data = json.loads(content)
    except Exception as e:
        if not isinstance(e, FileNotFoundError):
            raise e

    # Ensure existing data is a list
    if not isinstance(existing_data, list):
        existing_data = []

    # Append new data
    existing_data.extend(new_data)

    # Write back to file
    with open(file_path, 'w') as file:
        json.dump(existing_data, file, indent=2)

def get_totp():
    try:
        totp = pyotp.TOTP(os.getenv('ANGEL_ONE_TOTP_KEY'))
        otp = totp.now()
        # Calculate time remaining until next OTP
        remaining_seconds = totp.interval - datetime.now().timestamp() % totp.interval
        expiry_time = datetime.now() + timedelta(seconds=remaining_seconds)
        print(f'TOTP expires at {expiry_time.strftime("%H:%M:%S")}')
        return otp
    except Exception as e:
        print(f"Error generating TOTP: {e}")
        return None

def format_date(date):
    return date.strftime("%Y-%m-%d %H:%M")

def process_request(smart_api, symbol_token, from_date, to_date):
    try:
        response = smart_api.getCandleData({
            "exchange": "NSE",
            "symboltoken": symbol_token,
            "interval": "ONE_MINUTE",
            "fromdate": from_date,
            "todate": to_date
        })
        
        if response and isinstance(response, dict) and response.get('data'):
            append_json_to_file('data.json', response['data'])
            print(f'Saved data from {from_date} to {to_date}')
        else:
            print('Failed response')
            print(response)
    except Exception as e:
        print(f"Error in process_request: {e}")

def login_now(smart_api):
    try:
        login = smart_api.generateSession(
            os.getenv('ANGEL_ONE_CLIENT_ID'),
            os.getenv('ANGEL_ONE_CLIENT_PIN'),
            get_totp()
        )
        print(f"Session Generated: {login.get('status', False)}")
        return login.get('status', False)
    except Exception as e:
        print(f"Error in login: {e}")
        return False

def get_requests(smart_api):
    try:
        start_date = datetime.strptime('2018-03-27 12:00', '%Y-%m-%d %H:%M')
        end_date = datetime.strptime('2024-02-10 09:00', '%Y-%m-%d %H:%M')
        max_days = 30
        
        current_start = start_date
        i = 0
        
        while current_start < end_date:
            # Refresh login and TOTP every 3 iterations
            if i % 3 == 0:
                login_success = login_now(smart_api)
                if not login_success:
                    print("Failed to refresh login. Retrying in 10 seconds...")
                    time.sleep(10)
                    continue
            i += 1
            
            # 3 second delay to prevent rate limiting
            time.sleep(3)
            
            current_end = min(
                current_start + timedelta(days=max_days),
                end_date
            )
            
            time.sleep(3)
            process_request(
                smart_api,
                "14366",
                format_date(current_start),
                format_date(current_end)
            )
            
            # Add 1 minute to avoid overlap
            current_start = current_end + timedelta(minutes=1)
            
    except Exception as e:
        print(f"Error in get_requests: {e}")

def main():
    try:
        smart_api = SmartConnect(api_key=os.getenv('ANGEL_ONE_API_KEY'))
        
        if login_now(smart_api):
            # Uncomment to test a single request
            # process_request(smart_api, "14366", "2024-01-01 13:31", "2024-01-01 13:43")
            get_requests(smart_api)
        else:
            print("Initial login failed")
    except Exception as e:
        print(f"Error in main: {e}")

if __name__ == "__main__":
    main()