#[test]
fn dogfood() {
    if option_env!("RUSTC_TEST_SUITE").is_some() || cfg!(windows) {
        return;
    }
    let root_dir = std::env::current_dir().unwrap();
    for d in &[".", "clippy_lints", "rustc_tools_util", "clippy_dev"] {
        std::env::set_current_dir(root_dir.join(d)).unwrap();
        let output = std::process::Command::new("cargo")
            .arg("run")
            .arg("--bin")
            .arg("cargo-clippy")
            .arg("--all-features")
            .arg("--manifest-path")
            .arg(root_dir.join("Cargo.toml"))
            .args(&["--", "-W clippy::internal"])
            .env("CLIPPY_DOGFOOD", "true")
            .output()
            .unwrap();
        println!("status: {}", output.status);
        println!("stdout: {}", String::from_utf8_lossy(&output.stdout));
        println!("stderr: {}", String::from_utf8_lossy(&output.stderr));

        assert!(output.status.success());
    }
}
