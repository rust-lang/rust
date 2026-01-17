use std::path::PathBuf;

use clap::Parser;

use crate::diagnostics::TidyFlags;

#[test]
fn test_tidy_flags_parser() {
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
        "auto:js,auto:py,auto:cpp,auto:spellcheck",
        "--", // pos
        "some-file",
        "some-file2",
    ];
    let tidy_flags = TidyFlags::try_parse_from(args).unwrap();

    assert_eq!(tidy_flags.root_path, PathBuf::from("/home/user/rust"));
    assert_eq!(
        tidy_flags.cargo,
        PathBuf::from("/home/user/rust/build/x86_64-unknown-linux-gnu/stage0/bin/cargo")
    );
    assert_eq!(tidy_flags.output_directory, PathBuf::from("/home/user/rust/build"));
    assert_eq!(tidy_flags.concurrency, 16);
    assert_eq!(tidy_flags.npm, PathBuf::from("/home/user/rust/build/misc-tools/bin/yarn"));
    assert!(tidy_flags.verbose);
    assert!(tidy_flags.bless);
    assert_eq!(
        tidy_flags.extra_checks.unwrap(),
        vec![
            "auto:js".to_string(),
            "auto:py".to_string(),
            "auto:cpp".to_string(),
            "auto:spellcheck".to_string(),
        ]
    );
    assert_eq!(tidy_flags.pos, vec!["some-file".to_string(), "some-file2".to_string()]);
}
