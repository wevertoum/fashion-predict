import CloudUploadIcon from "@mui/icons-material/CloudUpload";
import {
	Alert,
	Box,
	Button,
	Card,
	CardContent,
	CardMedia,
	CircularProgress,
	Container,
	Input,
	Typography,
} from "@mui/material";
import type React from "react";
import { useState } from "react";

interface PredictionResult {
	predicted_class: string;
	confidence: number;
	probabilities: number[];
}

export default function FashionPredictLabPage() {
	const [selectedFile, setSelectedFile] = useState<File | null>(null);
	const [imagePreviewUrl, setImagePreviewUrl] = useState<string | null>(null);
	const [prediction, setPrediction] = useState<PredictionResult | null>(null);
	const [loading, setLoading] = useState(false);
	const [error, setError] = useState<string | null>(null);

	const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
		const file = event.target.files?.[0];
		if (file) {
			setSelectedFile(file);
			setImagePreviewUrl(URL.createObjectURL(file));
			setPrediction(null);
			setError(null);
		} else {
			setSelectedFile(null);
			setImagePreviewUrl(null);
		}
	};

	const handleSubmit = async () => {
		if (!selectedFile) {
			setError("Por favor, selecione uma imagem para classificar.");
			return;
		}

		setLoading(true);
		setError(null);
		setPrediction(null);

		const formData = new FormData();
		formData.append("file", selectedFile);

		try {
			const response = await fetch("http://127.0.0.1:5000/predict", {
				method: "POST",
				body: formData,
			});

			if (!response.ok) {
				const errorData = await response.json();
				throw new Error(
					errorData.error || "Erro desconhecido ao classificar a imagem.",
				);
			}

			const data = await response.json();
			setPrediction(data);
		} catch (err) {
			console.error("Erro ao fazer a requisição:", err);
			setError(
				`Falha ao conectar com o servidor ou erro de classificação: ${(err as Error).message}`,
			);
		} finally {
			setLoading(false);
		}
	};

	return (
		<Container maxWidth="sm" sx={{ mt: 8, mb: 4 }}>
			<Card sx={{ p: 4, borderRadius: 2, boxShadow: 3 }}>
				<CardContent sx={{ textAlign: "center" }}>
					<Typography
						variant="h4"
						component="h1"
						gutterBottom
						sx={{ color: "primary.main", fontWeight: "bold" }}
					>
						Fashion MNIST Classification Lab
					</Typography>

					<Box
						sx={{
							mb: 3,
							display: "flex",
							flexDirection: "column",
							alignItems: "center",
						}}
					>
						<Button
							variant="contained"
							component="label"
							startIcon={<CloudUploadIcon />}
							sx={{ mb: 2 }}
						>
							Selecionar Imagem
							<Input
								type="file"
								hidden
								inputProps={{ accept: "image/*" }}
								onChange={handleFileChange}
							/>
						</Button>

						{imagePreviewUrl && (
							<Box
								sx={{
									mt: 2,
									border: "1px solid #e0e0e0",
									borderRadius: 1,
									overflow: "hidden",
								}}
							>
								<CardMedia
									component="img"
									image={imagePreviewUrl}
									alt="Pré-visualização da Imagem"
									sx={{ maxWidth: 200, maxHeight: 200, display: "block" }}
								/>
							</Box>
						)}

						<Button
							variant="contained"
							color="success"
							onClick={handleSubmit}
							disabled={loading || !selectedFile}
							sx={{ mt: 3, py: 1.5, px: 4 }}
						>
							{loading ? (
								<CircularProgress size={24} color="inherit" />
							) : (
								"Classificar Imagem"
							)}
						</Button>
					</Box>

					{error && (
						<Alert severity="error" sx={{ mt: 3 }}>
							{error}
						</Alert>
					)}

					{prediction && (
						<Box sx={{ mt: 4, pt: 3, borderTop: "1px solid #e0e0e0" }}>
							<Typography
								variant="h5"
								component="h2"
								gutterBottom
								sx={{ color: "secondary.main", fontWeight: "medium" }}
							>
								Resultado da Classificação:
							</Typography>
							<Typography variant="h6" sx={{ mb: 1 }}>
								Previsão:{" "}
								<strong style={{ color: "#1976d2" }}>
									{prediction.predicted_class}
								</strong>
							</Typography>
							<Typography variant="body1">
								Confiança:{" "}
								<strong style={{ color: "#1976d2" }}>
									{(prediction.confidence * 100).toFixed(2)}%
								</strong>
							</Typography>
						</Box>
					)}
				</CardContent>
			</Card>
		</Container>
	);
}
