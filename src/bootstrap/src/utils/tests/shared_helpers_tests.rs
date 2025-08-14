//! The `shared_helpers` module can't have its own tests submodule, because that would cause
//! problems for the shim binaries that include it via `#[path]`, so instead those unit tests live
//! here.
//!
//! To prevent tidy from complaining about this file not being named `tests.rs`, it lives inside a
//! submodule directory named `tests`.

use crate::utils::shared_helpers::parse_value_from_args;

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
