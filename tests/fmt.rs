#![cfg_attr(feature = "deny-warnings", deny(warnings))]
#![warn(rust_2018_idioms, unused_lifetimes)]

use std::path::PathBuf;
use std::process::Command;

#[test]
fn fmt() {
    if option_env!("RUSTC_TEST_SUITE").is_some() || option_env!("NO_FMT_TEST").is_some() {
        return;
    }

    // Skip this test if nightly rustfmt is unavailable
    let rustup_output = Command::new("rustup")
        .args(&["component", "list", "--toolchain", "nightly"])
        .output()
        .unwrap();
    assert!(rustup_output.status.success());
    let component_output = String::from_utf8_lossy(&rustup_output.stdout);
    if !component_output.contains("rustfmt") {
        return;
    }

    let root_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let output = Command::new("cargo")
        .current_dir(root_dir)
        .args(&["dev", "fmt", "--check"])
        .output()
        .unwrap();

    println!("status: {}", output.status);
    println!("stdout: {}", String::from_utf8_lossy(&output.stdout));
    println!("stderr: {}", String::from_utf8_lossy(&output.stderr));

    assert!(
        output.status.success(),
        "Formatting check failed. Run `cargo dev fmt` to update formatting."
    );
}
