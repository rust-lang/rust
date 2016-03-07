#![feature(plugin)]
#![plugin(clippy, regex_macros)]

#![allow(unused)]
#![deny(invalid_regex, trivial_regex, regex_macro)]

extern crate regex;

fn main() {
    let some_regex = regex!("for real!"); //~ERROR `regex!(_)`
    let other_regex = regex!("[a-z]_[A-Z]"); //~ERROR `regex!(_)`
}
