#[test]
fn test_timeout() {
    let mut cmd = assert_cmd::cargo::cargo_bin_cmd!();
    cmd.env("TEST_DEVICE_CONNECT_TIMEOUT_SECONDS", "1");
    cmd.env("TEST_DEVICE_ADDR", "0.0.0.0:6969");
    cmd.args(["spawn-emulator", "riscv64-unknown-linux-gnu", "./"]);
    cmd.arg(std::env::temp_dir());

    let assert = cmd.assert().failure();
    let output = assert.get_output();

    let stderr = String::from_utf8(output.stderr.clone()).unwrap();
    assert!(stderr.contains("Gave up trying to connect to test device"));
}
