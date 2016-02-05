#![feature(plugin)]
#![plugin(clippy)]

#![allow(unused)]
#![deny(invalid_regex)]

extern crate regex;

use regex::Regex;

const OPENING_PAREN : &'static str = "(";

fn main() {
    let pipe_in_wrong_position = Regex::new("|");
    //~^ERROR: Regex syntax error: empty alternate
    let wrong_char_ranice = Regex::new("[z-a]"); 
    //~^ERROR: Regex syntax error: invalid character class range
    
    let some_regex = Regex::new(OPENING_PAREN);
    //~^ERROR: Regex syntax error on position 0: unclosed

    let closing_paren = ")";
    let not_linted = Regex::new(closing_paren);
}
