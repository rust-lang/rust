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
    if !contains(expected, &stdout) {
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

fn contains(expected: &str, actual: &str) -> bool {
    // Normalize for Windows paths.
    let actual = actual.replace('\\', "/");
    actual.lines().any(|mut line| {
        for (i, part) in expected.split("[..]").enumerate() {
            match line.find(part) {
                Some(j) => {
                    if i == 0 && j != 0 {
                        return false;
                    }
                    line = &line[j + part.len()..];
                }
                None => return false,
            }
        }
        line.is_empty() || expected.ends_with("[..]")
    })
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
    broken_test("basic_broken", "foo.html:3: broken link - `bar.html`");
}

#[test]
fn broken_fragment_local() {
    broken_test(
        "broken_fragment_local",
        "foo.html:3: broken link fragment `#somefrag` pointing to `foo.html`",
    );
}

#[test]
fn broken_fragment_remote() {
    broken_test(
        "broken_fragment_remote/inner",
        "foo.html:3: broken link fragment `#somefrag` pointing to \
         `[..]/broken_fragment_remote/bar.html`",
    );
}

#[test]
fn broken_redir() {
    broken_test(
        "broken_redir",
        "foo.html:3: broken redirect from `redir-bad.html` to `sometarget`",
    );
}

#[test]
fn directory_link() {
    broken_test(
        "directory_link",
        "foo.html:3: directory link to `somedir` (directory links should use index.html instead)",
    );
}

#[test]
fn redirect_loop() {
    broken_test(
        "redirect_loop",
        "foo.html:3: redirect from `redir-bad.html` to `[..]redirect_loop/redir-bad.html` \
         which is also a redirect (not supported)",
    );
}

#[test]
fn broken_intra_doc_link() {
    broken_test(
        "broken_intra_doc_link",
        "foo.html:3: broken intra-doc link - [<code>std::ffi::CString</code>]",
    );
}
