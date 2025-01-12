

const API_URL = "http://18.206.118.139:5001/chat"; 

document.getElementById("upload-form").addEventListener("submit", async (event) => {
    event.preventDefault();

    const status = document.getElementById("status");
    const searchTerm = document.getElementById("search-term").value;
    const fileInput = document.getElementById("file-upload");
    
    if (!searchTerm) {
        status.textContent = "Please enter a search term.";
        return;
    }

    status.textContent = "Sending message...";

    const formData = new FormData();
    formData.append("message", searchTerm);
    
    if (fileInput.files.length > 0) {
        formData.append("file", fileInput.files[0]);
    }

    try {
        const response = await fetch(API_URL, {
            method: "POST",
            body: formData
        });

        if (response.ok) {
            const result = await response.json();
            status.textContent = `Chatbot says: ${result.response}`;
            updateChat(result.response);  // Update chat container with the response
        } else {
            status.textContent = "Error communicating with chatbot.";
        }
    } catch (error) {
        console.error("Error:", error);
        status.textContent = "Failed to connect to server.";
    }
});

function updateChat(response) {
    const chatContainer = document.getElementById("chat-container");
    const messageDiv = document.createElement("div");
    messageDiv.textContent = `Bot: ${response}`;
    chatContainer.appendChild(messageDiv);
}

});