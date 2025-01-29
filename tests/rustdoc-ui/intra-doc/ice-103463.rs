// https://github.com/rust-lang/rust/issues/103463
//@ check-pass

// The `Trait` is not pulled into the crate resulting in doc links in its methods being resolved.

//@ aux-build:issue-103463-aux.rs

extern crate issue_103463_aux;
use issue_103463_aux::Trait;

fn main() {}
