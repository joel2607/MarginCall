let { SmartAPI, WebSocket,WebSocketV2 } = require('smartapi-javascript');
const { TOTP  } = require('totp-generator')
require('dotenv').config();
const fs = require('fs').promises;

async function appendJsonToFile(filePath, newData) {
    let existingData = [];

    try {
        // Check if file exists
        await fs.access(filePath);
        
        // Read existing data
        const fileContent = await fs.readFile(filePath, 'utf-8');
        if (fileContent.trim()) {
            existingData = JSON.parse(fileContent);
        }
    } catch (error) {
        // Ignore "file not found" error, throw others
        if (error.code !== 'ENOENT') {
            throw error;
        }
    }

    // Ensure existing data is an array
    if (!Array.isArray(existingData)) {
        existingData = [];
    }

    // Append new data
    existingData.push(...newData);

    // Write back to file
    await fs.writeFile(filePath, JSON.stringify(existingData, null, 2), 'utf-8');
}
const getTOTP = () => {
    try {

        const {otp, expires} = TOTP.generate(process.env.ANGEL_ONE_TOTP_KEY);
        console.log('TOTP expires at ' + new Date(expires).toLocaleTimeString());
        return otp;
    } catch (err) {
        console.log(err)
    }
}
function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

function formatDate(date) {
    return date.toISOString().slice(0, 16).replace('T', ' ');
}

let smart_api = new SmartAPI({
	api_key: process.env.ANGEL_ONE_API_KEY, 
});

const processRequest = async (symbolToken, from, to) => {
    const response = await smart_api.getCandleData({
        exchange: "NSE",
        symboltoken: symbolToken,
        interval: "ONE_MINUTE",
        fromdate: from,
        todate: to
    });
    if(response.status === true && response.data.length !== 0){
        appendJsonToFile('data.json', response.data);
        console.log(`Saved data from ${from} to ${to}`);     
    }
    else{
        
        console.log('Failed response');
        console.log(response)
    }
}
const loginNow = async () => {

    const login = await smart_api.generateSession(process.env.ANGEL_ONE_CLIENT_ID, process.env.ANGEL_ONE_CLIENT_PIN, getTOTP())
    console.log(`Session Generated: ${login.status}`);
}

const getRequests = async () => {
    try {
        const startDate = new Date('2014-02-10T09:00:00');
        const endDate = new Date('2024-02-10T09:00:00');
        const maxDaysPerRequest = 30;

        let currentStart = new Date(startDate);
        let i = 0;
        while (currentStart < endDate) {

            //5 second delay to prevent getting ratelimited
            await sleep(5000);

            //Refresh login and TOTP 
            if(i %3 == 0) await loginNow();
            i++;

            const currentEnd = new Date(Math.min(currentStart.getTime() + maxDaysPerRequest * 24 * 60 * 60 * 1000, endDate.getTime()));
            
            processRequest("14366", formatDate(currentStart), formatDate(currentEnd));
            
            currentStart = new Date(currentEnd.getTime() + 60000); // Add 1 minute to avoid overlap
        } 
    }   
    catch (err) {
        console.log(err);
    }
}

const main = async () => {

    await loginNow();
    //await processRequest("14366", "2014-04-11 03:32", "2014-05-11 03:32")
    await getRequests();
}

main();
