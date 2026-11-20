#[test]
fn test_help() {
    let mut cmd = assert_cmd::cargo::cargo_bin_cmd!();
    cmd.arg("help");

    let output = cmd.unwrap();

    let stdout = String::from_utf8(output.stdout.clone()).unwrap();
    assert!(stdout.trim().starts_with("Usage:"));
}

#[test]
fn test_timeout() {
    let mut cmd = assert_cmd::cargo::cargo_bin_cmd!();
    cmd.env("TEST_DEVICE_CONNECT_TIMEOUT_SECONDS", "1");
    cmd.env("TEST_DEVICE_ADDR", "127.69.69.69:6969");
    cmd.args(["spawn-emulator", "dummy-target", "dummy-server", "dummy-tmpdir"]);

    let assert = cmd.assert().failure();
    let output = assert.get_output();

    let stderr = String::from_utf8(output.stderr.clone()).unwrap();
    let pass_msg = "Gave up trying to connect to test device";
    assert!(stderr.contains(pass_msg), "Could not find `{pass_msg}` in `{stderr}`");
}
