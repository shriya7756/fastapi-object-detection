document.getElementById("start-btn").addEventListener("click", function () {
    if (confirm("Allow access to your webcam for object detection?")) {
        document.getElementById("video-container").classList.remove("hidden");
        document.getElementById("video-feed").src = "/video_feed"; // Start the video stream
    }
});

document.getElementById("know-more-btn").addEventListener("click", function () {
    let descriptionBox = document.getElementById("description-box");
    if (descriptionBox.style.display === "none" || descriptionBox.classList.contains("hidden")) {
        descriptionBox.style.display = "block";
    } else {
        descriptionBox.style.display = "none";
    }
});
