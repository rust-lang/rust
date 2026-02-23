use std::path::PathBuf;

use crate::arg_parser::TidyArgParser;

// Test all arguments
#[test]
fn test_tidy_parser_full() {
    let args = vec![
        "rust-tidy",
        "--root-path",
        "/home/user/rust",
        "--cargo-path",
        "/home/user/rust/build/x86_64-unknown-linux-gnu/stage0/bin/cargo",
        "--output-dir",
        "/home/user/rust/build",
        "--concurrency",
        "16",
        "--npm-path",
        "yarn",
        "--verbose",
        "--bless",
        "--extra-checks",
        "if-installed:auto:js,auto:if-installed:py,if-installed:auto:cpp,if-installed:auto:spellcheck",
        "--", // pos_args
        "some-file",
        "some-file2",
    ];
    let cmd = TidyArgParser::command();
    let parsed_args = TidyArgParser::build(cmd.get_matches_from(args));

    assert_eq!(parsed_args.root_path, PathBuf::from("/home/user/rust"));
    assert_eq!(
        parsed_args.cargo,
        PathBuf::from("/home/user/rust/build/x86_64-unknown-linux-gnu/stage0/bin/cargo")
    );
    assert_eq!(parsed_args.output_directory, PathBuf::from("/home/user/rust/build"));
    assert_eq!(parsed_args.concurrency.get(), 16);
    assert_eq!(parsed_args.npm, PathBuf::from("yarn"));
    assert!(parsed_args.verbose);
    assert!(parsed_args.bless);
    assert_eq!(
        parsed_args.extra_checks,
        Some(vec![
            "if-installed:auto:js".to_string(),
            "auto:if-installed:py".to_string(),
            "if-installed:auto:cpp".to_string(),
            "if-installed:auto:spellcheck".to_string(),
        ])
    );
    assert_eq!(parsed_args.pos_args, vec!["some-file".to_string(), "some-file2".to_string()]);
}

// The parser can take required args any order
#[test]
fn test_tidy_parser_any_order() {
    let args = vec![
        "rust-tidy",
        "--npm-path",
        "yarn",
        "--concurrency",
        "16",
        "--output-dir",
        "/home/user/rust/build",
        "--cargo-path",
        "/home/user/rust/build/x86_64-unknown-linux-gnu/stage0/bin/cargo",
        "--root-path",
        "/home/user/rust",
    ];
    let cmd = TidyArgParser::command();
    let parsed_args = TidyArgParser::build(cmd.get_matches_from(args));

    assert_eq!(parsed_args.root_path, PathBuf::from("/home/user/rust"));
    assert_eq!(
        parsed_args.cargo,
        PathBuf::from("/home/user/rust/build/x86_64-unknown-linux-gnu/stage0/bin/cargo")
    );
    assert_eq!(parsed_args.output_directory, PathBuf::from("/home/user/rust/build"));
    assert_eq!(parsed_args.concurrency.get(), 16);
    assert_eq!(parsed_args.npm, PathBuf::from("yarn"));
}

// --root-path is required
#[test]
fn test_tidy_parser_missing_root_path() {
    let args = vec![
        "rust-tidy",
        "--npm-path",
        "yarn",
        "--concurrency",
        "16",
        "--output-dir",
        "/home/user/rust/build",
        "--cargo-path",
        "/home/user/rust/build/x86_64-unknown-linux-gnu/stage0/bin/cargo",
    ];
    let cmd = TidyArgParser::command();
    assert!(cmd.try_get_matches_from(args).is_err());
}

// --cargo-path is required
#[test]
fn test_tidy_parser_missing_cargo_path() {
    let args = vec![
        "rust-tidy",
        "--npm-path",
        "yarn",
        "--concurrency",
        "16",
        "--output-dir",
        "/home/user/rust/build",
        "--root-path",
        "/home/user/rust",
    ];
    let cmd = TidyArgParser::command();
    assert!(cmd.try_get_matches_from(args).is_err());
}

// --output-dir is required
#[test]
fn test_tidy_parser_missing_output_dir() {
    let args = vec![
        "rust-tidy",
        "--npm-path",
        "yarn",
        "--concurrency",
        "16",
        "--cargo-path",
        "/home/user/rust/build/x86_64-unknown-linux-gnu/stage0/bin/cargo",
        "--root-path",
        "/home/user/rust",
    ];
    let cmd = TidyArgParser::command();
    assert!(cmd.try_get_matches_from(args).is_err());
}

// --concurrency is required
#[test]
fn test_tidy_parser_missing_concurrency() {
    let args = vec![
        "rust-tidy",
        "--npm-path",
        "yarn",
        "--output-dir",
        "/home/user/rust/build",
        "--cargo-path",
        "/home/user/rust/build/x86_64-unknown-linux-gnu/stage0/bin/cargo",
        "--root-path",
        "/home/user/rust",
    ];
    let cmd = TidyArgParser::command();
    assert!(cmd.try_get_matches_from(args).is_err());
}

// --npm-path is required
#[test]
fn test_tidy_parser_missing_npm_path() {
    let args = vec![
        "rust-tidy",
        "--concurrency",
        "16",
        "--output-dir",
        "/home/user/rust/build",
        "--cargo-path",
        "/home/user/rust/build/x86_64-unknown-linux-gnu/stage0/bin/cargo",
    ];
    let cmd = TidyArgParser::command();
    assert!(cmd.try_get_matches_from(args).is_err());
}
