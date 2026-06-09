//@ check-pass
//@ edition:2018
//@ compile-flags: --extern issue_56596_2
//@ aux-build:issue-56596-2.rs

mod m {
    use core::any;
    pub use issue_56596_2::*;
}

fn main() {}
