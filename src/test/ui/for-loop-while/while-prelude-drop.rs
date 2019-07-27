// run-pass
#![allow(non_camel_case_types)]

use std::string::String;

#[derive(PartialEq)]
enum t { a, b(String), }

fn make(i: isize) -> t {
    if i > 10 { return t::a; }
    let mut s = String::from("hello");
    // Ensure s is non-const.

    s.push_str("there");
    return t::b(s);
}

pub fn main() {
    let mut i = 0;


    // The auto slot for the result of make(i) should not leak.
    while make(i) != t::a { i += 1; }
}
