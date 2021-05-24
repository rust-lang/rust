use std::path::Path;
use std::process::{Command, ExitStatus};

fn run(dirname: &str) -> (ExitStatus, String, String) {
    let output = Command::new(env!("CARGO_BIN_EXE_linkchecker"))
        .current_dir(Path::new(env!("CARGO_MANIFEST_DIR")).join("tests"))
        .arg(dirname)
        .output()
        .unwrap();
    let stdout = String::from_utf8(output.stdout).unwrap();
    let stderr = String::from_utf8(output.stderr).unwrap();
    (output.status, stdout, stderr)
}

fn broken_test(dirname: &str, expected: &str) {
    let (status, stdout, stderr) = run(dirname);
    assert!(!status.success());
    if !stdout.contains(expected) {
        panic!(
            "stdout did not contain expected text: {}\n\
            --- stdout:\n\
            {}\n\
            --- stderr:\n\
            {}\n",
            expected, stdout, stderr
        );
    }
}

fn valid_test(dirname: &str) {
    let (status, stdout, stderr) = run(dirname);
    if !status.success() {
        panic!(
            "test did not succeed as expected\n\
            --- stdout:\n\
            {}\n\
            --- stderr:\n\
            {}\n",
            stdout, stderr
        );
    }
}

#[test]
fn valid() {
    valid_test("valid/inner");
}

#[test]
fn basic_broken() {
    broken_test("basic_broken", "bar.html");
}

#[test]
fn broken_fragment_local() {
    broken_test("broken_fragment_local", "#somefrag");
}

#[test]
fn broken_fragment_remote() {
    broken_test("broken_fragment_remote/inner", "#somefrag");
}

#[test]
fn broken_redir() {
    broken_test("broken_redir", "sometarget");
}

#[test]
fn directory_link() {
    broken_test("directory_link", "somedir");
}

#[test]
fn redirect_loop() {
    broken_test("redirect_loop", "redir-bad.html");
}
