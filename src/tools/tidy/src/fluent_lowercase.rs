//! Checks that the error messages start with a lowercased letter (except when allowed to).

use std::path::Path;

use fluent_syntax::ast::{Entry, Message, PatternElement};

use crate::walk::{filter_dirs, walk};

#[rustfmt::skip]
const ALLOWED_CAPITALIZED_WORDS: &[&str] = &[
    // tidy-alphabetical-start
    "ABI",
    "ABIs",
    "ADT",
    "C",
    "CGU",
    "Ferris",
    "MIR",
    "OK",
    "Rust",
    "VS", // VS Code
    // tidy-alphabetical-end
];

fn filter_fluent(path: &Path) -> bool {
    if let Some(ext) = path.extension() { ext.to_str() != Some("ftl") } else { true }
}

fn is_allowed_capitalized_word(msg: &str) -> bool {
    ALLOWED_CAPITALIZED_WORDS.iter().any(|word| {
        msg.strip_prefix(word)
            .map(|tail| tail.chars().next().map(|c| c == '-' || c.is_whitespace()).unwrap_or(true))
            .unwrap_or_default()
    })
}

fn check_lowercase(filename: &str, contents: &str, bad: &mut bool) {
    let (Ok(parse) | Err((parse, _))) = fluent_syntax::parser::parse(contents);

    for entry in &parse.body {
        if let Entry::Message(msg) = entry
            && let Message { value: Some(pattern), .. } = msg
            && let [first_pattern, ..] = &pattern.elements[..]
            && let PatternElement::TextElement { value } = first_pattern
            && value.chars().next().is_some_and(char::is_uppercase)
            && !is_allowed_capitalized_word(value)
        {
            tidy_error!(
                bad,
                "{filename}: message `{value}` starts with an uppercase letter. Fix it or add it to `ALLOWED_CAPITALIZED_WORDS`"
            );
        }
    }
}

pub fn check(path: &Path, bad: &mut bool) {
    walk(
        path,
        |path, is_dir| filter_dirs(path) || (!is_dir && filter_fluent(path)),
        &mut |ent, contents| {
            check_lowercase(ent.path().to_str().unwrap(), contents, bad);
        },
    );
}
