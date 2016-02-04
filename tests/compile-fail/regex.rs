#![feature(plugin)]
#![plugin(clippy)]

#![allow(unused)]
#![deny(invalid_regex)]

extern crate regex;

use regex::Regex;

fn main() {
    let pipe_in_wrong_position = Regex::new("|");
    //~^ERROR: Regex syntax error: empty alternate
    let wrong_char_range = Regex::new("[z-a]"); 
    //~^ERROR: Regex syntax error: invalid character class range
}
