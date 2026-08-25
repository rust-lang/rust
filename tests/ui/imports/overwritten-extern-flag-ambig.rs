// Test for issue #145575.

//@ check-pass
//@ edition: 2018

extern crate core as std;

mod inner {
    use crate::*;
    use std::str; // OK for now
}

fn main() {}
