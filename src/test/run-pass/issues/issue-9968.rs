// run-pass
// aux-build:issue-9968.rs

// pretty-expanded FIXME #23616

extern crate issue_9968 as lib;

use lib::{Trait, Struct};

pub fn main()
{
    Struct::init().test();
}
