import json
import pandas as pd

# Open and read the JSON file
with open('C:\\Users\\Joel\\projects\\MarginCall\\data-collection\\data.json', 'r') as f:
    data = json.load(f)

# Create DataFrame with specified headers
df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

# Save to CSV
df.to_csv('new.csv', index=False)

# Verify the contents
print(df.head())
print("\nJSON file successfully converted to CSV!")