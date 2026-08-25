//@ edition:2018
//@ compile-flags: --extern issue_56596
//@ aux-build:issue-56596.rs

mod m {
    pub mod issue_56596 {}
}

use m::*;
use issue_56596; //~ ERROR `issue_56596` is ambiguous

fn main() {}
