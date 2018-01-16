#[test]
fn dogfood() {
    if option_env!("RUSTC_TEST_SUITE").is_some() {
        return;
    }
    if cfg!(windows) {
        return;
    }
    let root_dir = std::env::current_dir().unwrap();
    for d in &[".", "clippy_lints"] {
        std::env::set_current_dir(root_dir.join(d)).unwrap();
        let output = std::process::Command::new("cargo")
            .arg("run")
            .arg("--bin").arg("cargo-clippy")
            .arg("--manifest-path").arg(root_dir.join("Cargo.toml"))
            .output().unwrap();
        println!("status: {}", output.status);
        println!("stdout: {}", String::from_utf8_lossy(&output.stdout));
        println!("stderr: {}", String::from_utf8_lossy(&output.stderr));

        assert!(output.status.success());
    }
}
