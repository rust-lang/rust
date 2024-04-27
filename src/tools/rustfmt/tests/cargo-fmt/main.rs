// Integration tests for cargo-fmt.

use std::env;
use std::path::Path;
use std::process::Command;

use rustfmt_config_proc_macro::rustfmt_only_ci_test;

/// Run the cargo-fmt executable and return its output.
fn cargo_fmt(args: &[&str]) -> (String, String) {
    let mut bin_dir = env::current_exe().unwrap();
    bin_dir.pop(); // chop off test exe name
    if bin_dir.ends_with("deps") {
        bin_dir.pop();
    }
    let cmd = bin_dir.join(format!("cargo-fmt{}", env::consts::EXE_SUFFIX));

    // Ensure cargo-fmt runs the rustfmt binary from the local target dir.
    let path = env::var_os("PATH").unwrap_or_default();
    let mut paths = env::split_paths(&path).collect::<Vec<_>>();
    paths.insert(0, bin_dir);
    let new_path = env::join_paths(paths).unwrap();

    match Command::new(&cmd).args(args).env("PATH", new_path).output() {
        Ok(output) => (
            String::from_utf8(output.stdout).expect("utf-8"),
            String::from_utf8(output.stderr).expect("utf-8"),
        ),
        Err(e) => panic!("failed to run `{cmd:?} {args:?}`: {e}"),
    }
}

macro_rules! assert_that {
    ($args:expr, $check:ident $check_args:tt) => {
        let (stdout, stderr) = cargo_fmt($args);
        if !stdout.$check$check_args {
            panic!(
                "Output not expected for cargo-fmt {:?}\n\
                 expected: {}{}\n\
                 actual stdout:\n{}\n\
                 actual stderr:\n{}",
                $args,
                stringify!($check),
                stringify!($check_args),
                stdout,
                stderr
            );
        }
    };
}

#[rustfmt_only_ci_test]
#[test]
fn version() {
    assert_that!(&["--version"], starts_with("rustfmt "));
    assert_that!(&["--version"], starts_with("rustfmt "));
    assert_that!(&["--", "-V"], starts_with("rustfmt "));
    assert_that!(&["--", "--version"], starts_with("rustfmt "));
}

#[rustfmt_only_ci_test]
#[test]
fn print_config() {
    assert_that!(
        &["--", "--print-config", "current", "."],
        contains("max_width = ")
    );
}

#[rustfmt_only_ci_test]
#[test]
fn rustfmt_help() {
    assert_that!(&["--", "--help"], contains("Format Rust code"));
    assert_that!(&["--", "-h"], contains("Format Rust code"));
    assert_that!(&["--", "--help=config"], contains("Configuration Options:"));
}

#[rustfmt_only_ci_test]
#[test]
fn cargo_fmt_out_of_line_test_modules() {
    // See also https://github.com/rust-lang/rustfmt/issues/5119
    let expected_modified_files = [
        "tests/mod-resolver/test-submodule-issue-5119/src/lib.rs",
        "tests/mod-resolver/test-submodule-issue-5119/tests/test1.rs",
        "tests/mod-resolver/test-submodule-issue-5119/tests/test1/sub1.rs",
        "tests/mod-resolver/test-submodule-issue-5119/tests/test1/sub2.rs",
        "tests/mod-resolver/test-submodule-issue-5119/tests/test1/sub3/sub4.rs",
    ];
    let args = [
        "-v",
        "--check",
        "--manifest-path",
        "tests/mod-resolver/test-submodule-issue-5119/Cargo.toml",
    ];
    let (stdout, _) = cargo_fmt(&args);
    for file in expected_modified_files {
        let path = Path::new(file).canonicalize().unwrap();
        assert!(stdout.contains(&format!("Diff in {}", path.display())))
    }
}

#[rustfmt_only_ci_test]
#[test]
fn cargo_fmt_emits_error_on_line_overflow_true() {
    // See also https://github.com/rust-lang/rustfmt/issues/3164
    let args = [
        "--check",
        "--manifest-path",
        "tests/cargo-fmt/source/issue_3164/Cargo.toml",
        "--",
        "--config",
        "error_on_line_overflow=true",
    ];

    let (_stdout, stderr) = cargo_fmt(&args);
    assert!(stderr.contains(
        "line formatted, but exceeded maximum width (maximum: 100 (see `max_width` option)"
    ))
}
