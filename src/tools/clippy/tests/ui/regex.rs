//@require-annotations-for-level: WARN
#![allow(
    unused,
    clippy::needless_raw_strings,
    clippy::needless_raw_string_hashes,
    clippy::needless_borrow,
    clippy::needless_borrows_for_generic_args
)]
#![warn(clippy::invalid_regex, clippy::trivial_regex, clippy::regex_creation_in_loops)]

extern crate regex;

use regex::bytes::{Regex as BRegex, RegexBuilder as BRegexBuilder, RegexSet as BRegexSet};
use regex::{Regex, RegexBuilder, RegexSet};

const OPENING_PAREN: &str = "(";
const NOT_A_REAL_REGEX: &str = "foobar";

fn syntax_error() {
    let pipe_in_wrong_position = Regex::new("|");
    //~^ ERROR: trivial regex
    let pipe_in_wrong_position_builder = RegexBuilder::new("|");
    //~^ ERROR: trivial regex
    let wrong_char_ranice = Regex::new("[z-a]");
    //~^ ERROR: regex syntax error: invalid character class range, the start must be <= th
    //~| NOTE: `-D clippy::invalid-regex` implied by `-D warnings`
    let some_unicode = Regex::new("[é-è]");
    //~^ ERROR: regex syntax error: invalid character class range, the start must be <= th

    let some_regex = Regex::new(OPENING_PAREN);
    //~^ invalid_regex

    let binary_pipe_in_wrong_position = BRegex::new("|");
    //~^ ERROR: trivial regex
    let some_binary_regex = BRegex::new(OPENING_PAREN);
    //~^ invalid_regex
    let some_binary_regex_builder = BRegexBuilder::new(OPENING_PAREN);
    //~^ invalid_regex

    let closing_paren = ")";
    let not_linted = Regex::new(closing_paren);

    let set = RegexSet::new(&[r"[a-z]+@[a-z]+\.(com|org|net)", r"[a-z]+\.(com|org|net)"]);
    let bset = BRegexSet::new(&[
        r"[a-z]+@[a-z]+\.(com|org|net)",
        r"[a-z]+\.(com|org|net)",
        r".", // regression test
    ]);

    let set_error = RegexSet::new(&[OPENING_PAREN, r"[a-z]+\.(com|org|net)"]);
    //~^ invalid_regex
    let bset_error = BRegexSet::new(&[OPENING_PAREN, r"[a-z]+\.(com|org|net)"]);
    //~^ invalid_regex

    // These following three cases are considering valid since regex-1.8.0
    let raw_string_error = Regex::new(r"[...\/...]");
    let raw_string_error = Regex::new(r#"[...\/...]"#);
    let _ = Regex::new(r"(?<hi>hi)").unwrap();

    let escaped_string_span = Regex::new("\\b\\c");
    //~^ invalid_regex

    let aux_span = Regex::new("(?ixi)");
    //~^ ERROR: regex syntax error: duplicate flag

    let should_not_lint = Regex::new("(?u).");
    let should_not_lint = BRegex::new("(?u).");
    let invalid_utf8_should_not_lint = BRegex::new("(?-u).");
    let invalid_utf8_should_lint = Regex::new("(?-u).");
    //~^ ERROR: regex syntax error: pattern can match invalid UTF-8
}

fn trivial_regex() {
    let trivial_eq = Regex::new("^foobar$");
    //~^ ERROR: trivial regex

    let trivial_eq_builder = RegexBuilder::new("^foobar$");
    //~^ ERROR: trivial regex

    let trivial_starts_with = Regex::new("^foobar");
    //~^ ERROR: trivial regex

    let trivial_ends_with = Regex::new("foobar$");
    //~^ ERROR: trivial regex

    let trivial_contains = Regex::new("foobar");
    //~^ ERROR: trivial regex

    let trivial_contains = Regex::new(NOT_A_REAL_REGEX);
    //~^ ERROR: trivial regex

    let trivial_backslash = Regex::new("a\\.b");
    //~^ ERROR: trivial regex

    // unlikely corner cases
    let trivial_empty = Regex::new("");
    //~^ ERROR: trivial regex

    let trivial_empty = Regex::new("^");
    //~^ ERROR: trivial regex

    let trivial_empty = Regex::new("^$");
    //~^ ERROR: trivial regex

    let binary_trivial_empty = BRegex::new("^$");
    //~^ ERROR: trivial regex

    // non-trivial regexes
    let non_trivial_dot = Regex::new("a.b");
    let non_trivial_dot_builder = RegexBuilder::new("a.b");
    let non_trivial_dot = Regex::new(".");
    let non_trivial_dot = BRegex::new(".");
    let non_trivial_eq = Regex::new("^foo|bar$");
    let non_trivial_starts_with = Regex::new("^foo|bar");
    let non_trivial_ends_with = Regex::new("^foo|bar");
    let non_trivial_ends_with = Regex::new("foo|bar");
    let non_trivial_binary = BRegex::new("foo|bar");
    let non_trivial_binary_builder = BRegexBuilder::new("foo|bar");

    // #6005: unicode classes in bytes::Regex
    let a_byte_of_unicode = BRegex::new(r"\p{C}");

    // start and end word boundary, introduced in regex 0.10
    let _ = BRegex::new(r"\<word\>");
    let _ = BRegex::new(r"\b{start}word\b{end}");
}

fn regex_creation_in_loops() {
    loop {
        static STATIC_REGEX: std::sync::LazyLock<Regex> = std::sync::LazyLock::new(|| Regex::new("a.b").unwrap());

        let regex = Regex::new("a.b");
        //~^ ERROR: compiling a regex in a loop
        let regex = BRegex::new("a.b");
        //~^ ERROR: compiling a regex in a loop
        #[allow(clippy::regex_creation_in_loops)]
        let allowed_regex = Regex::new("a.b");

        if true {
            let regex = Regex::new("a.b");
            //~^ ERROR: compiling a regex in a loop
        }

        for _ in 0..10 {
            let nested_regex = Regex::new("a.b");
            //~^ ERROR: compiling a regex in a loop
        }
    }

    for i in 0..10 {
        let dependant_regex = Regex::new(&format!("{i}"));
    }
}

fn main() {
    syntax_error();
    trivial_regex();
    regex_creation_in_loops();
}
