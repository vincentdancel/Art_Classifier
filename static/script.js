class ArtClassifier {
  constructor() {
    this.mediaStream = null;
    this.canvas = document.getElementById("photoCanvas");
    this.ctx = this.canvas.getContext("2d");
    this.video = document.getElementById("cameraFeed");
    this.imageBlob = null;

    this.initEventListeners();
    this.loadArtists();
  }

  initEventListeners() {
    const uploadBox = document.getElementById("uploadBox");

    // FILE INPUT
    document.getElementById("fileInput").addEventListener("change", (e) => {
      this.handleFileSelect(e);
    });

    // DRAG OVER (REQUIRED)
    uploadBox.addEventListener("dragover", (e) => {
      e.preventDefault();
      uploadBox.classList.add("drag-over");
    });

    // DRAG LEAVE
    uploadBox.addEventListener("dragleave", () => {
      uploadBox.classList.remove("drag-over");
    });

    // DROP
    uploadBox.addEventListener("drop", (e) => {
      e.preventDefault();
      uploadBox.classList.remove("drag-over");
      const file = e.dataTransfer.files[0];
      if (!file || !file.type.startsWith("image/")) {
        alert("Please drop an image file");
        return;
      }
      this.handleDroppedFile(file);
    });

    document.getElementById("cameraBtn").addEventListener("click", () => {
      this.startCamera();
    });

    document.getElementById("stopCameraBtn").addEventListener("click", () => {
      this.stopCamera();
    });

    document.getElementById("captureBtn").addEventListener("click", () => {
      this.capturePhoto();
    });

    document.getElementById("analyzeBtn").addEventListener("click", () => {
      this.analyzeImage();
    });

    document.getElementById("clearBtn").addEventListener("click", () => {
      this.clearImage();
    });
  }

  // =========================
  // CAMERA (LOW RES, FAST)
  // =========================
  async startCamera() {
    try {
      this.mediaStream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: "environment",
          width: { ideal: 640 },
          height: { ideal: 640 },
        },
      });

      document.getElementById("uploadBox").style.display = "none";
      document.getElementById("cameraContainer").style.display = "block";
      document.getElementById("previewContainer").style.display = "none";

      this.video.srcObject = this.mediaStream;
      await this.video.play();
    } catch (err) {
      alert("Camera error: " + err.message);
    }
  }

  stopCamera() {
    if (this.mediaStream) {
      this.mediaStream.getTracks().forEach((track) => track.stop());
      this.mediaStream = null;
    }

    this.imageBlob = null;

    document.getElementById("cameraContainer").style.display = "none";
    document.getElementById("previewContainer").style.display = "none";
    document.getElementById("uploadBox").style.display = "block";

    this.video.srcObject = null;
  }

  // =========================
  // CAPTURE → 160x160
  // =========================
  capturePhoto() {
    if (!this.mediaStream) return;

    // Set canvas size
    this.canvas.width = 160;
    this.canvas.height = 160;

    // Draw video frame to canvas
    this.ctx.drawImage(this.video, 0, 0, 160, 160);

    // Convert to blob for analysis
    this.canvas.toBlob(
      (blob) => {
        this.imageBlob = blob;
        document.getElementById("previewImage").src = URL.createObjectURL(blob);

        document.getElementById("cameraContainer").style.display = "none";
        document.getElementById("previewContainer").style.display = "block";
      },
      "image/jpeg",
      0.95
    );
  }

  // =========================
  // FILE UPLOAD → 160x160
  // =========================
  handleFileSelect(event) {
    const file = event.target.files[0];
    if (!file) return;

    const img = new Image();
    img.onload = () => {
      const normalizedCanvas = this.normalizeImage(img);

      normalizedCanvas.toBlob(
        (blob) => {
          this.imageBlob = blob;
          document.getElementById("previewImage").src =
            URL.createObjectURL(blob);

          document.getElementById("uploadBox").style.display = "none";
          document.getElementById("previewContainer").style.display = "block";
        },
        "image/jpeg",
        0.85
      );
    };
    img.src = URL.createObjectURL(file);
  }

  handleDroppedFile(file) {
    this.handleFileSelect({ target: { files: [file] } });
  }

  // =========================
  // NORMALIZE IMAGE
  // =========================
  normalizeImage(imageElement) {
    const canvas = document.createElement("canvas");
    canvas.width = 160;
    canvas.height = 160;
    const ctx = canvas.getContext("2d");

    ctx.fillStyle = "#FFFFFF";
    ctx.fillRect(0, 0, 160, 160);

    const ratio = Math.min(160 / imageElement.width, 160 / imageElement.height);
    const newWidth = imageElement.width * ratio;
    const newHeight = imageElement.height * ratio;

    const x = (160 - newWidth) / 2;
    const y = (160 - newHeight) / 2;

    ctx.drawImage(imageElement, x, y, newWidth, newHeight);

    return canvas;
  }

  // =========================
  // ANALYZE IMAGE
  // =========================
  async analyzeImage() {
    if (!this.imageBlob) {
      alert("Select or capture an image first");
      return;
    }

    document.getElementById("results").style.display = "none";
    document.getElementById("initialPlaceholder").style.display = "none";
    document.getElementById("loading").style.display = "block";

    try {
      const formData = new FormData();
      formData.append("file", this.imageBlob, "image.jpg");

      const response = await fetch("/predict", {
        method: "POST",
        body: formData,
      });

      const result = await response.json();
      if (!result.success) throw new Error(result.error || "Prediction failed");

      this.displayResults(result);
    } catch (err) {
      alert("Analysis error: " + err.message);
      console.error(err);

      document.getElementById("loading").style.display = "none";
      document.getElementById("initialPlaceholder").style.display = "block";
    }
  }

  clearImage() {
    this.imageBlob = null;
    document.getElementById("previewContainer").style.display = "none";
    document.getElementById("uploadBox").style.display = "block";
    this.stopCamera();
  }

  // =========================
  // DISPLAY RESULTS
  // =========================
  displayResults(result) {
    const p = result.prediction;

    document.getElementById("initialPlaceholder").style.display = "none";
    document.getElementById("loading").style.display = "none";
    document.getElementById("results").style.display = "block";

    document.getElementById("artistName").textContent = p.artist.replace(
      /_/g,
      " "
    );
    document.getElementById("confidenceBadge").textContent = `${(
      p.confidence * 100
    ).toFixed(1)}%`;
    document.getElementById("artistGenre").textContent = Array.isArray(p.genre)
      ? p.genre.join(", ")
      : p.genre;
    document.getElementById("artistNationality").textContent = p.nationality;
    document.getElementById("artistYears").textContent = p.years;
    document.getElementById("artistHistory").textContent = p.history;

    const worksList = document.getElementById("famousWorksList");
    worksList.innerHTML = "";
    if (p.famous_works && p.famous_works.length > 0) {
      p.famous_works.forEach((work) => {
        const li = document.createElement("li");
        li.textContent = work;
        worksList.appendChild(li);
      });
    } else {
      worksList.innerHTML = "<li>No works listed</li>";
    }

    const predictionList = document.getElementById("predictionList");
    predictionList.innerHTML = "";
    if (result.top_predictions && result.top_predictions.length > 0) {
      result.top_predictions.forEach((pred) => {
        const div = document.createElement("div");
        div.className = "prediction-item";
        div.innerHTML = `
          <span>${pred.artist.replace(/_/g, " ")}</span>
          <span>${(pred.confidence * 100).toFixed(1)}%</span>
        `;
        predictionList.appendChild(div);
      });
    }
  }

  async loadArtists() {
    try {
      const res = await fetch("/status");
      const data = await res.json();
      console.log("Artists loaded:", data.artists);
    } catch {}
  }
}

document.addEventListener("DOMContentLoaded", () => {
  new ArtClassifier();
});
