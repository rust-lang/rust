#![deny(unknown_lints)]

#![allow(not_a_real_lint)] //~ ERROR unknown lint

#![deny(dead_cod)] //~ ERROR unknown lint
                   //~| HELP did you mean
                   //~| SUGGESTION dead_code

fn main() {}
