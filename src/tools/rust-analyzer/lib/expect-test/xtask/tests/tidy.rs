use xaction::cmd;

#[test]
fn test_formatting() {
    cmd!("cargo fmt --all -- --check").run().unwrap()
}
