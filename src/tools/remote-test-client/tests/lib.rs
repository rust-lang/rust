#[test]
fn test_help() {
    let mut cmd = assert_cmd::cargo::cargo_bin_cmd!();
    cmd.arg("help");

    let output = cmd.unwrap();

    let stdout = String::from_utf8(output.stdout.clone()).unwrap();
    assert!(stdout.trim().starts_with("Usage:"));
}
