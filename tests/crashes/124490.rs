//@ known-bug: rust-lang/rust#124490
use io::{self as std};
use std::collections::{self as io};

mod a {
    pub mod b {
        pub mod c {}
    }
}

use a::*;

use b::c;
use c as b;

fn main() {}
