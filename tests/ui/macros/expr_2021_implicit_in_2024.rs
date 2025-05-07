//@ edition: 2024
//@ aux-build:expr_2021_implicit.rs

//@ check-pass

extern crate expr_2021_implicit;

// Makes sure that a `:expr` fragment matcher defined in a edition 2021 crate
// still parses like an `expr_2021` fragment matcher in a 2024 user crate.
expr_2021_implicit::m!(const {});

fn main() {}
