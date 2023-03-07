#![deny(unknown_lints)]

#![allow(not_a_real_lint)] //~ ERROR unknown lint

#![deny(dead_cod)] //~ ERROR unknown lint
                   //~| HELP did you mean
                   //~| SUGGESTION dead_code

#![deny(rust_2018_idiots)] //~ ERROR unknown lint
                           //~| HELP did you mean
                           //~| SUGGESTION rust_2018_idioms

fn main() {}
