fn main() {
    if std::env::args().any(|v| v == "-vV") {
        std::process::Command::new(std::env::var("RUSTC_REAL").unwrap())
            .arg("-vV")
            .status()
            .unwrap();
    } else {
        todo!("rustc-shim")
    }
}
