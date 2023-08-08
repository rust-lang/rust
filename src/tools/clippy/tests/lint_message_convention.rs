#![feature(lazy_cell)]
#![cfg_attr(feature = "deny-warnings", deny(warnings))]
#![warn(rust_2018_idioms, unused_lifetimes)]

use std::ffi::OsStr;
use std::path::PathBuf;
use std::sync::LazyLock;

use regex::RegexSet;

#[derive(Debug)]
struct Message {
    path: PathBuf,
    bad_lines: Vec<String>,
}

impl Message {
    fn new(path: PathBuf) -> Self {
        // we don't want the first letter after "error: ", "help: " ... to be capitalized
        // also no punctuation (except for "?" ?) at the end of a line
        // Prefer "try" over "try this".
        static REGEX_SET: LazyLock<RegexSet> = LazyLock::new(|| {
            RegexSet::new([
                "error: [A-Z]",
                "help: [A-Z]",
                "warning: [A-Z]",
                "note: [A-Z]",
                "try: [A-Z]",
                "error: .*[.!]$",
                "help: .*[.!]$",
                "warning: .*[.!]$",
                "note: .*[.!]$",
                "try: .*[.!]$",
                "try this",
            ])
            .unwrap()
        });

        // sometimes the first character is capitalized and it is legal (like in "C-like enum variants") or
        // we want to ask a question ending in "?"
        static EXCEPTIONS_SET: LazyLock<RegexSet> = LazyLock::new(|| {
            RegexSet::new([
                r"\.\.\.$",
                ".*C-like enum variant discriminant is not portable to 32-bit targets",
                ".*Intel x86 assembly syntax used",
                ".*AT&T x86 assembly syntax used",
                "note: Clippy version: .*",
                "the compiler unexpectedly panicked. this is a bug.",
            ])
            .unwrap()
        });

        let content: String = std::fs::read_to_string(&path).unwrap();

        let bad_lines = content
            .lines()
            .filter(|line| REGEX_SET.matches(line).matched_any())
            // ignore exceptions
            .filter(|line| !EXCEPTIONS_SET.matches(line).matched_any())
            .map(ToOwned::to_owned)
            .collect::<Vec<String>>();

        Message { path, bad_lines }
    }
}

#[test]
fn lint_message_convention() {
    // disable the test inside the rustc test suite
    if option_env!("RUSTC_TEST_SUITE").is_some() {
        return;
    }

    // make sure that lint messages:
    // * are not capitalized
    // * don't have punctuation at the end of the last sentence

    // these directories have interesting tests
    let test_dirs = ["ui", "ui-cargo", "ui-internal", "ui-toml"]
        .iter()
        .map(PathBuf::from)
        .map(|p| {
            let base = PathBuf::from("tests");
            base.join(p)
        });

    // gather all .stderr files
    let tests = test_dirs
        .flat_map(|dir| {
            std::fs::read_dir(dir)
                .expect("failed to read dir")
                .map(|direntry| direntry.unwrap().path())
        })
        .filter(|file| matches!(file.extension().map(OsStr::to_str), Some(Some("stderr"))));

    // get all files that have any "bad lines" in them
    let bad_tests: Vec<Message> = tests
        .map(Message::new)
        .filter(|message| !message.bad_lines.is_empty())
        .collect();

    for message in &bad_tests {
        eprintln!(
            "error: the test '{}' contained the following nonconforming lines :",
            message.path.display()
        );
        message.bad_lines.iter().for_each(|line| eprintln!("{line}"));
        eprintln!("\n\n");
    }

    eprintln!(
        "\n\n\nLint message should not start with a capital letter and should not have punctuation at the end of the message unless multiple sentences are needed."
    );
    eprintln!("Check out the rustc-dev-guide for more information:");
    eprintln!("https://rustc-dev-guide.rust-lang.org/diagnostics.html#diagnostic-structure\n\n\n");

    assert!(bad_tests.is_empty());
}
