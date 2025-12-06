//@ aux-build: issue-114682-4-extern.rs
// https://github.com/rust-lang/rust/pull/114682#issuecomment-1880755441

extern crate issue_114682_4_extern;

use issue_114682_4_extern::*;

//~v ERROR type alias takes 1 generic argument but 2 generic arguments were supplied
fn a() -> Result<i32, ()> { //~ ERROR `Result` is ambiguous
                            //~| WARN this was previously accepted
    Ok(1)
}

fn main() {}
