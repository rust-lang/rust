#[test]
fn dogfood() {
    if option_env!("RUSTC_TEST_SUITE").is_some() || cfg!(windows) {
        return;
    }
    let root_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let clippy_cmd = std::path::Path::new(&root_dir)
        .join("target")
        .join(env!("PROFILE"))
        .join("cargo-clippy");

    let output = std::process::Command::new(clippy_cmd)
        .current_dir(root_dir)
        .env("CLIPPY_DOGFOOD", "1")
        .arg("clippy-preview")
        .arg("--all-targets")
        .arg("--all-features")
        .arg("--")
        .args(&["-D", "clippy::all"])
        .args(&["-D", "clippy::internal"])
        .args(&["-D", "clippy::pedantic"])
        .output()
        .unwrap();
    println!("status: {}", output.status);
    println!("stdout: {}", String::from_utf8_lossy(&output.stdout));
    println!("stderr: {}", String::from_utf8_lossy(&output.stderr));

    assert!(output.status.success());
}

#[test]
fn dogfood_tests() {
    if option_env!("RUSTC_TEST_SUITE").is_some() || cfg!(windows) {
        return;
    }
    let root_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let clippy_cmd = std::path::Path::new(&root_dir)
        .join("target")
        .join(env!("PROFILE"))
        .join("cargo-clippy");

    for d in &[
        "clippy_workspace_tests",
        "clippy_workspace_tests/src",
        "clippy_workspace_tests/subcrate",
        "clippy_workspace_tests/subcrate/src",
        "clippy_dev",
        "rustc_tools_util",
    ] {
        let output = std::process::Command::new(&clippy_cmd)
            .current_dir(root_dir.join(d))
            .env("CLIPPY_DOGFOOD", "1")
            .arg("clippy")
            .arg("--")
            .args(&["-D", "clippy::all"])
            .args(&["-D", "clippy::pedantic"])
            .output()
            .unwrap();
        println!("status: {}", output.status);
        println!("stdout: {}", String::from_utf8_lossy(&output.stdout));
        println!("stderr: {}", String::from_utf8_lossy(&output.stderr));

        assert!(output.status.success());
    }
}
