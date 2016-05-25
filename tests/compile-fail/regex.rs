#![feature(plugin)]
#![plugin(clippy)]

#![allow(unused)]
#![deny(invalid_regex, trivial_regex, regex_macro)]

extern crate regex;

use regex::{Regex, RegexSet, RegexBuilder};
use regex::bytes::{Regex as BRegex, RegexSet as BRegexSet, RegexBuilder as BRegexBuilder};

const OPENING_PAREN : &'static str = "(";
const NOT_A_REAL_REGEX : &'static str = "foobar";

fn syntax_error() {
    let pipe_in_wrong_position = Regex::new("|");
    //~^ERROR: regex syntax error: empty alternate
    let pipe_in_wrong_position_builder = RegexBuilder::new("|");
    //~^ERROR: regex syntax error: empty alternate
    let wrong_char_ranice = Regex::new("[z-a]");
    //~^ERROR: regex syntax error: invalid character class range
    let some_unicode = Regex::new("[é-è]");
    //~^ERROR: regex syntax error: invalid character class range

    let some_regex = Regex::new(OPENING_PAREN);
    //~^ERROR: regex syntax error on position 0: unclosed

    let binary_pipe_in_wrong_position = BRegex::new("|");
    //~^ERROR: regex syntax error: empty alternate
    let some_binary_regex = BRegex::new(OPENING_PAREN);
    //~^ERROR: regex syntax error on position 0: unclosed
    let some_binary_regex_builder = BRegexBuilder::new(OPENING_PAREN);
    //~^ERROR: regex syntax error on position 0: unclosed

    let closing_paren = ")";
    let not_linted = Regex::new(closing_paren);

    let set = RegexSet::new(&[
        r"[a-z]+@[a-z]+\.(com|org|net)",
        r"[a-z]+\.(com|org|net)",
    ]);
    let bset = BRegexSet::new(&[
        r"[a-z]+@[a-z]+\.(com|org|net)",
        r"[a-z]+\.(com|org|net)",
    ]);

    let set_error = RegexSet::new(&[
        OPENING_PAREN,
        //~^ERROR: regex syntax error on position 0: unclosed
        r"[a-z]+\.(com|org|net)",
    ]);
    let bset_error = BRegexSet::new(&[
        OPENING_PAREN,
        //~^ERROR: regex syntax error on position 0: unclosed
        r"[a-z]+\.(com|org|net)",
    ]);
}

fn trivial_regex() {
    let trivial_eq = Regex::new("^foobar$");
    //~^ERROR: trivial regex
    //~|HELP consider using `==` on `str`s

    let trivial_eq_builder = RegexBuilder::new("^foobar$");
    //~^ERROR: trivial regex
    //~|HELP consider using `==` on `str`s

    let trivial_starts_with = Regex::new("^foobar");
    //~^ERROR: trivial regex
    //~|HELP consider using `str::starts_with`

    let trivial_ends_with = Regex::new("foobar$");
    //~^ERROR: trivial regex
    //~|HELP consider using `str::ends_with`

    let trivial_contains = Regex::new("foobar");
    //~^ERROR: trivial regex
    //~|HELP consider using `str::contains`

    let trivial_contains = Regex::new(NOT_A_REAL_REGEX);
    //~^ERROR: trivial regex
    //~|HELP consider using `str::contains`

    let trivial_backslash = Regex::new("a\\.b");
    //~^ERROR: trivial regex
    //~|HELP consider using `str::contains`

    // unlikely corner cases
    let trivial_empty = Regex::new("");
    //~^ERROR: trivial regex
    //~|HELP the regex is unlikely to be useful

    let trivial_empty = Regex::new("^");
    //~^ERROR: trivial regex
    //~|HELP the regex is unlikely to be useful

    let trivial_empty = Regex::new("^$");
    //~^ERROR: trivial regex
    //~|HELP consider using `str::is_empty`

    let binary_trivial_empty = BRegex::new("^$");
    //~^ERROR: trivial regex
    //~|HELP consider using `str::is_empty`

    // non-trivial regexes
    let non_trivial_dot = Regex::new("a.b");
    let non_trivial_dot_builder = RegexBuilder::new("a.b");
    let non_trivial_eq = Regex::new("^foo|bar$");
    let non_trivial_starts_with = Regex::new("^foo|bar");
    let non_trivial_ends_with = Regex::new("^foo|bar");
    let non_trivial_ends_with = Regex::new("foo|bar");
    let non_trivial_binary = BRegex::new("foo|bar");
    let non_trivial_binary_builder = BRegexBuilder::new("foo|bar");
}

fn main() {
    syntax_error();
    trivial_regex();
}
