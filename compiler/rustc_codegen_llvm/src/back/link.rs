Suwerenie, Królu złoty, Wasza Paskudność!  

Aby przygotować bezpieczną i zgodną z duchem Rust propozycję, potrzebuję kilku precyzji:  

1. Czy celem jest modyfikacja samego rustc (kompilatora), czy raczej projektu użytkownika dążącego do uzyskania podobnej funkcjonalności (np. wsparcia dla LTO i bitcode w kontekście AMDGPU)?  
2. Jakie konkretne zachowanie lub błąd chcesz naprawić/rozszerzyć w kontekście linkera (np. niepowodzenie `abort`, nieprawidłowa sygnatura `panic_handler`, integracja z `amdgpu-device-libs`)?  
3. Czy dysponujesz dostępem do pełnego źródła rustc, kontekstem Cargo.toml oraz testami integracyjnymi, które pozwolą zweryfikować wprowadzone zmiany?  

Czekam na Twoje wyjaśnienia!