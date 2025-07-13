//@ run-pass
//@ aux-build:pub-use-link-errors-issue-9968.rs


extern crate pub_use_link_errors_issue_9968 as lib;

use lib::{Trait, Struct};

pub fn main()
{
    Struct::init().test();
}

// https://github.com/rust-lang/rust/issues/9968
