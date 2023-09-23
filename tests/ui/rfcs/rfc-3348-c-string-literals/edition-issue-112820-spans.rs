// edition: 2021
// known-bug: #112820
//
// aux-build: count.rs
#![feature(c_str_literals)]

// aux-build: wrong_parsing.rs
extern crate wrong_parsing;

const _: () = {
    wrong_parsing::repro!(c"cstr");
};

fn main() {}
