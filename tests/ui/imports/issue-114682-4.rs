//@ check-pass
//@ aux-build: issue-114682-4-extern.rs
// https://github.com/rust-lang/rust/pull/114682#issuecomment-1880755441

extern crate issue_114682_4_extern;

use issue_114682_4_extern::*;

fn a() -> Result<i32, ()> { // FIXME: `Result` should be identified as an ambiguous item.
    Ok(1)
}

fn main() {}
