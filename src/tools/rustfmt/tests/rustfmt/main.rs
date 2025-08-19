//! Integration tests for rustfmt.

use std::env;
use std::fs::remove_file;
use std::path::Path;
use std::process::Command;

use rustfmt_config_proc_macro::{nightly_only_test, rustfmt_only_ci_test};

/// Run the rustfmt executable and return its output.
fn rustfmt(args: &[&str]) -> (String, String) {
    let mut bin_dir = env::current_exe().unwrap();
    bin_dir.pop(); // chop off test exe name
    if bin_dir.ends_with("deps") {
        bin_dir.pop();
    }
    let cmd = bin_dir.join(format!("rustfmt{}", env::consts::EXE_SUFFIX));

    // Ensure the rustfmt binary runs from the local target dir.
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
    ($args:expr, $($check:ident $check_args:tt)&&+) => {
        let (stdout, stderr) = rustfmt($args);
        if $(!stdout.$check$check_args && !stderr.$check$check_args)||* {
            panic!(
                "Output not expected for rustfmt {:?}\n\
                 expected: {}\n\
                 actual stdout:\n{}\n\
                 actual stderr:\n{}",
                $args,
                stringify!($( $check$check_args )&&*),
                stdout,
                stderr
            );
        }
    };
}

#[rustfmt_only_ci_test]
#[test]
fn print_config() {
    assert_that!(
        &["--print-config", "unknown"],
        starts_with("Unknown print-config option")
    );
    assert_that!(&["--print-config", "default"], contains("max_width = 100"));
    assert_that!(&["--print-config", "minimal"], contains("PATH required"));
    assert_that!(
        &["--print-config", "minimal", "minimal-config"],
        contains("doesn't work with standard input.")
    );

    let (stdout, stderr) = rustfmt(&[
        "--print-config",
        "minimal",
        "minimal-config",
        "src/shape.rs",
    ]);
    assert!(
        Path::new("minimal-config").exists(),
        "stdout:\n{stdout}\nstderr:\n{stderr}"
    );
    remove_file("minimal-config").unwrap();
}

#[rustfmt_only_ci_test]
#[test]
fn inline_config() {
    // single invocation
    assert_that!(
        &[
            "--print-config",
            "current",
            ".",
            "--config=color=Never,edition=2018"
        ],
        contains("color = \"Never\"") && contains("edition = \"2018\"")
    );

    // multiple overriding invocations
    assert_that!(
        &[
            "--print-config",
            "current",
            ".",
            "--config",
            "color=never,edition=2018",
            "--config",
            "color=always,format_strings=true"
        ],
        contains("color = \"Always\"")
            && contains("edition = \"2018\"")
            && contains("format_strings = true")
    );
}

#[test]
fn rustfmt_usage_text() {
    let args = ["--help"];
    let (stdout, _) = rustfmt(&args);
    assert!(stdout.contains("Format Rust code\n\nusage: rustfmt [options] <file>..."));
}

#[test]
fn mod_resolution_error_multiple_candidate_files() {
    // See also https://github.com/rust-lang/rustfmt/issues/5167
    let default_path = Path::new("tests/mod-resolver/issue-5167/src/a.rs");
    let secondary_path = Path::new("tests/mod-resolver/issue-5167/src/a/mod.rs");
    let error_message = format!(
        "file for module found at both {:?} and {:?}",
        default_path.canonicalize().unwrap(),
        secondary_path.canonicalize().unwrap(),
    );

    let args = ["tests/mod-resolver/issue-5167/src/lib.rs"];
    let (_stdout, stderr) = rustfmt(&args);
    assert!(stderr.contains(&error_message))
}

#[test]
fn mod_resolution_error_sibling_module_not_found() {
    let args = ["tests/mod-resolver/module-not-found/sibling_module/lib.rs"];
    let (_stdout, stderr) = rustfmt(&args);
    // Module resolution fails because we're unable to find `a.rs` in the same directory as lib.rs
    assert!(stderr.contains("a.rs does not exist"))
}

#[test]
fn mod_resolution_error_relative_module_not_found() {
    let args = ["tests/mod-resolver/module-not-found/relative_module/lib.rs"];
    let (_stdout, stderr) = rustfmt(&args);
    // The file `./a.rs` and directory `./a` both exist.
    // Module resolution fails because we're unable to find `./a/b.rs`
    #[cfg(not(windows))]
    assert!(stderr.contains("a/b.rs does not exist"));
    #[cfg(windows)]
    assert!(stderr.contains("a\\b.rs does not exist"));
}

#[test]
fn mod_resolution_error_path_attribute_does_not_exist() {
    let args = ["tests/mod-resolver/module-not-found/bad_path_attribute/lib.rs"];
    let (_stdout, stderr) = rustfmt(&args);
    // The path attribute points to a file that does not exist
    assert!(stderr.contains("does_not_exist.rs does not exist"));
}

#[test]
fn rustfmt_emits_error_on_line_overflow_true() {
    // See also https://github.com/rust-lang/rustfmt/issues/3164
    let args = [
        "--config",
        "error_on_line_overflow=true",
        "tests/cargo-fmt/source/issue_3164/src/main.rs",
    ];

    let (_stdout, stderr) = rustfmt(&args);
    assert!(stderr.contains(
        "line formatted, but exceeded maximum width (maximum: 100 (see `max_width` option)"
    ))
}

#[test]
#[allow(non_snake_case)]
fn dont_emit_ICE() {
    let files = [
        "tests/target/issue_5728.rs",
        "tests/target/issue_5729.rs",
        "tests/target/issue-5885.rs",
        "tests/target/issue_6082.rs",
        "tests/target/issue_6069.rs",
        "tests/target/issue-6105.rs",
    ];

    let panic_re = regex::Regex::new("thread.*panicked").unwrap();
    for file in files {
        let args = [file];
        let (_stdout, stderr) = rustfmt(&args);
        assert!(!panic_re.is_match(&stderr));
    }
}

#[test]
fn rustfmt_emits_error_when_control_brace_style_is_always_next_line() {
    // See also https://github.com/rust-lang/rustfmt/issues/5912
    let args = [
        "--config=color=Never",
        "--config",
        "control_brace_style=AlwaysNextLine",
        "--config",
        "match_arm_blocks=false",
        "tests/target/issue_5912.rs",
    ];

    let (_stdout, stderr) = rustfmt(&args);
    assert!(!stderr.contains("error[internal]: left behind trailing whitespace"))
}

#[nightly_only_test]
#[test]
fn rustfmt_generates_no_error_if_failed_format_code_in_doc_comments() {
    // See also https://github.com/rust-lang/rustfmt/issues/6109

    let file = "tests/target/issue-6109.rs";
    let args = ["--config", "format_code_in_doc_comments=true", file];
    let (stdout, stderr) = rustfmt(&args);
    assert!(stderr.is_empty());
    assert!(stdout.is_empty());
}

#[test]
fn rustfmt_error_improvement_regarding_invalid_toml() {
    // See also https://github.com/rust-lang/rustfmt/issues/6302
    let invalid_toml_config = "tests/config/issue-6302.toml";
    let args = ["--config-path", invalid_toml_config];
    let (_stdout, stderr) = rustfmt(&args);

    let toml_path = Path::new(invalid_toml_config).canonicalize().unwrap();
    let expected_error_message = format!("The file `{}` failed to parse", toml_path.display());

    assert!(stderr.contains(&expected_error_message));
}
