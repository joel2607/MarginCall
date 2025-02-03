let { SmartAPI, WebSocket,WebSocketV2 } = require('smartapi-javascript');
const { TOTP  } = require('totp-generator')
require('dotenv').config();

const fs = require('fs');

function appendJsonToFile(filePath, newData) {
    let existingData = [];

    // Read existing data if file exists
    if (fs.existsSync(filePath)) {
        const fileContent = fs.readFileSync(filePath, 'utf-8');
        if (fileContent.trim()) {
            existingData = JSON.parse(fileContent);
        }
    }

    // Ensure it's an array, then prepend new data
    if (!Array.isArray(existingData)) {
        existingData = [existingData];
    }
    existingData.unshift(newData); // Append on top

    // Write back to file
    fs.writeFileSync(filePath, JSON.stringify(existingData, null, 2), 'utf-8');
}

let smart_api = new SmartAPI({
	api_key: process.env.ANGEL_ONE_API_KEY, 
});


const getTOTP = () => {
    try {
        const {otp, expires} = TOTP.generate(process.env.ANGEL_ONE_TOTP_KEY);
        console.log('TOTP expires at ' + expires);
        return otp;
    } catch (err) {
        console.log(err)
    }
}


smart_api
	.generateSession(process.env.ANGEL_ONE_CLIENT_ID, process.env.ANGEL_ONE_CLIENT_PIN, getTOTP())
	.then((data) => {
		

		// User Methods
		// return smart_api.getProfile()

		// return smart_api.logout()

		//Historical Methods
		return smart_api.getCandleData({
		    "exchange": "NSE",
		    "symboltoken": "14366",
		    "interval": "ONE_MINUTE",
		    "fromdate": "2021-02-10 09:00",
		    "todate": "2022-02-10 09:20"
		})
	})
	.then((response) => {
		appendJsonToFile('data.json', response.data)
	})
	.catch((ex) => {
		console.log(ex)
	});

// // TO HANDLE SESSION EXPIRY, USERS CAN PROVIDE A CUSTOM FUNCTION AS PARAMETER TO setSessionExpiryHook METHOD
// smart_api.setSessionExpiryHook(customSessionHook);

// function customSessionHook() {
// 	console.log('User loggedout');

// 	// NEW AUTHENTICATION CAN TAKE PLACE HERE
// }