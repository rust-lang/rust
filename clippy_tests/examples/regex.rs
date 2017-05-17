#![feature(plugin)]
#![plugin(clippy)]

#![allow(unused)]
#![warn(invalid_regex, trivial_regex, regex_macro)]

extern crate regex;

use regex::{Regex, RegexSet, RegexBuilder};
use regex::bytes::{Regex as BRegex, RegexSet as BRegexSet, RegexBuilder as BRegexBuilder};

const OPENING_PAREN : &'static str = "(";
const NOT_A_REAL_REGEX : &'static str = "foobar";

fn syntax_error() {
    let pipe_in_wrong_position = Regex::new("|");
    let pipe_in_wrong_position_builder = RegexBuilder::new("|");
    let wrong_char_ranice = Regex::new("[z-a]");
    let some_unicode = Regex::new("[é-è]");

    let some_regex = Regex::new(OPENING_PAREN);

    let binary_pipe_in_wrong_position = BRegex::new("|");
    let some_binary_regex = BRegex::new(OPENING_PAREN);
    let some_binary_regex_builder = BRegexBuilder::new(OPENING_PAREN);

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
        r"[a-z]+\.(com|org|net)",
    ]);
    let bset_error = BRegexSet::new(&[
        OPENING_PAREN,
        r"[a-z]+\.(com|org|net)",
    ]);
}

fn trivial_regex() {
    let trivial_eq = Regex::new("^foobar$");

    let trivial_eq_builder = RegexBuilder::new("^foobar$");

    let trivial_starts_with = Regex::new("^foobar");

    let trivial_ends_with = Regex::new("foobar$");

    let trivial_contains = Regex::new("foobar");

    let trivial_contains = Regex::new(NOT_A_REAL_REGEX);

    let trivial_backslash = Regex::new("a\\.b");

    // unlikely corner cases
    let trivial_empty = Regex::new("");

    let trivial_empty = Regex::new("^");

    let trivial_empty = Regex::new("^$");

    let binary_trivial_empty = BRegex::new("^$");

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
