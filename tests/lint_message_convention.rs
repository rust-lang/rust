#![cfg_attr(feature = "deny-warnings", deny(warnings))]
#![warn(rust_2018_idioms, unused_lifetimes)]

use std::ffi::OsStr;
use std::path::PathBuf;

use regex::RegexSet;

#[derive(Debug)]
struct Message {
    path: PathBuf,
    bad_lines: Vec<String>,
}

impl Message {
    fn new(path: PathBuf) -> Self {
        let content: String = std::fs::read_to_string(&path).unwrap();
        // we don't want the first letter after "error: ", "help: " ... to be capitalized
        // also no puncutation (except for "?" ?) at the end of a line
        let regex_set: RegexSet = RegexSet::new(&[
            r"error: [A-Z]",
            r"help: [A-Z]",
            r"warning: [A-Z]",
            r"note: [A-Z]",
            r"try this: [A-Z]",
            r"error: .*[.!]$",
            r"help: .*[.!]$",
            r"warning: .*[.!]$",
            r"note: .*[.!]$",
            r"try this: .*[.!]$",
        ])
        .unwrap();

        // sometimes the first character is capitalized and it is legal (like in "C-like enum variants") or
        // we want to ask a question ending in "?"
        let exceptions_set: RegexSet = RegexSet::new(&[
            r".*C-like enum variant discriminant is not portable to 32-bit targets",
            r".*did you mean `unix`?",
            r".*the arguments may be inverted...",
            r".*Intel x86 assembly syntax used",
            r".*AT&T x86 assembly syntax used",
            r".*remove .*the return type...",
            r"note: Clippy version: .*",
            r"the compiler unexpectedly panicked. this is a bug.",
        ])
        .unwrap();

        let bad_lines = content
            .lines()
            .filter(|line| regex_set.matches(line).matched_any())
            // ignore exceptions
            .filter(|line| !exceptions_set.matches(line).matched_any())
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
    // * don't have puncuation at the end of the last sentence

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
        message.bad_lines.iter().for_each(|line| eprintln!("{}", line));
        eprintln!("\n\n");
    }

    eprintln!(
        "\n\n\nLint message should not start with a capital letter and should not have punctuation at the end of the message unless multiple sentences are needed."
    );
    eprintln!("Check out the rustc-dev-guide for more information:");
    eprintln!("https://rustc-dev-guide.rust-lang.org/diagnostics.html#diagnostic-structure\n\n\n");

    assert!(bad_tests.is_empty());
}
