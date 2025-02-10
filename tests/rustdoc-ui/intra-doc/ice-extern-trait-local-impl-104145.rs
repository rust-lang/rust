// https://github.com/rust-lang/rust/issues/104145
//@ check-pass

// Doc links in `Trait`'s methods are resolved because it has a local impl.

//@ aux-build:issue-103463-aux.rs

extern crate issue_103463_aux;
use issue_103463_aux::Trait;

pub struct LocalType;

impl Trait for LocalType {
    fn method() {}
}

fn main() {}
