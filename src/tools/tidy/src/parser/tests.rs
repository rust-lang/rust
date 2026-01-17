use std::path::PathBuf;

use crate::parser::TidyParser;

#[test]
fn test_tidy_parser() {
    let args = vec![
        "rust-tidy",
        "/home/user/rust", // Root dir
        "/home/user/rust/build/x86_64-unknown-linux-gnu/stage0/bin/cargo", // Cardo location
        "/home/user/rust/build", // Build dir
        "16",              // Number of concurrency
        "/home/user/rust/build/misc-tools/bin/yarn", // Yarn location
        "--verbose",
        "--bless",
        "--extra-checks",
        "if-installed:auto:js,auto:if-installed:py,if-installed:auto:cpp,if-installed:auto:spellcheck",
        "--", // pos
        "some-file",
        "some-file2",
    ];
    let cmd = TidyParser::command();
    let tidy_flags = TidyParser::build(cmd.get_matches_from(args));

    assert_eq!(tidy_flags.root_path, PathBuf::from("/home/user/rust"));
    assert_eq!(
        tidy_flags.cargo,
        PathBuf::from("/home/user/rust/build/x86_64-unknown-linux-gnu/stage0/bin/cargo")
    );
    assert_eq!(tidy_flags.output_directory, PathBuf::from("/home/user/rust/build"));
    assert_eq!(tidy_flags.concurrency.get(), 16);
    assert_eq!(tidy_flags.npm, PathBuf::from("/home/user/rust/build/misc-tools/bin/yarn"));
    assert!(tidy_flags.verbose);
    assert!(tidy_flags.bless);
    assert_eq!(
        tidy_flags.extra_checks,
        Some(vec![
            "if-installed:auto:js".to_string(),
            "auto:if-installed:py".to_string(),
            "if-installed:auto:cpp".to_string(),
            "if-installed:auto:spellcheck".to_string(),
        ])
    );
    assert_eq!(tidy_flags.pos, vec!["some-file".to_string(), "some-file2".to_string()]);
}
