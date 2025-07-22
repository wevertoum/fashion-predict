import { createTheme } from "@mui/material/styles";

export const theme = createTheme({
	palette: {
		mode: "light",
		primary: {
			main: "#1976d2",
		},
		secondary: {
			main: "#dc004e",
		},
		background: {
			default: "#f4f6f8",
			paper: "#ffffff",
		},
	},
	typography: {
		fontFamily: "Roboto, sans-serif",
		h1: {
			fontSize: "2.5rem",
			fontWeight: 600,
		},
	},
});
