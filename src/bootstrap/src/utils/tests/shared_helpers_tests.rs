//! The `shared_helpers` module can't have its own tests submodule, because that would cause
//! problems for the shim binaries that include it via `#[path]`, so instead those unit tests live
//! here.
//!
//! To prevent tidy from complaining about this file not being named `tests.rs`, it lives inside a
//! submodule directory named `tests`.

use std::ffi::OsString;
use std::fs;

use crate::utils::shared_helpers::{collect_args_from, parse_value_from_args};

#[test]
fn test_parse_value_from_args() {
    let args = vec![
        "--stage".into(),
        "1".into(),
        "--version".into(),
        "2".into(),
        "--target".into(),
        "x86_64-unknown-linux".into(),
    ];

    assert_eq!(parse_value_from_args(args.as_slice(), "--stage").unwrap(), "1");
    assert_eq!(parse_value_from_args(args.as_slice(), "--version").unwrap(), "2");
    assert_eq!(parse_value_from_args(args.as_slice(), "--target").unwrap(), "x86_64-unknown-linux");
    assert!(parse_value_from_args(args.as_slice(), "random-key").is_none());

    let args = vec![
        "app-name".into(),
        "--key".into(),
        "value".into(),
        "random-value".into(),
        "--sysroot=/x/y/z".into(),
    ];
    assert_eq!(parse_value_from_args(args.as_slice(), "--key").unwrap(), "value");
    assert_eq!(parse_value_from_args(args.as_slice(), "--sysroot").unwrap(), "/x/y/z");
}

#[test]
fn test_collect_args_expands_shell_argfile_enabled_in_argfile() {
    let tempdir = tempfile::tempdir().unwrap();
    let enable_shell_argfiles = tempdir.path().join("enable-shell-argfiles.args");
    let shell_argfile = tempdir.path().join("shell-argfiles.args");
    fs::write(&enable_shell_argfiles, "-Zshell-argfiles\n").unwrap();
    fs::write(&shell_argfile, "--target 'x86_64-unknown-linux-gnu' --cfg shell_argfile\n").unwrap();

    let args = vec![
        format!("@{}", enable_shell_argfiles.display()).into(),
        format!("@shell:{}", shell_argfile.display()).into(),
    ];

    assert_eq!(
        collect_args_from(args),
        ["-Zshell-argfiles", "--target", "x86_64-unknown-linux-gnu", "--cfg", "shell_argfile"]
            .map(OsString::from)
    );
}

#[test]
fn test_collect_args_expands_shell_argfile_enabled_by_separate_option() {
    let tempdir = tempfile::tempdir().unwrap();
    let shell_argfile = tempdir.path().join("shell-argfiles.args");
    fs::write(&shell_argfile, "--target 'x86_64-unknown-linux-gnu'\n").unwrap();

    let args = vec![
        "-Z".into(),
        "shell-argfiles".into(),
        format!("@shell:{}", shell_argfile.display()).into(),
    ];

    assert_eq!(
        collect_args_from(args),
        ["-Z", "shell-argfiles", "--target", "x86_64-unknown-linux-gnu"].map(OsString::from)
    );
}
