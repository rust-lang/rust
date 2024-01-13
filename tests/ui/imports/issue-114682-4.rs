// aux-build: issue-114682-4-extern.rs

// https://github.com/rust-lang/rust/pull/114682#issuecomment-1880755441
extern crate issue_114682_4_extern;

use issue_114682_4_extern::*;

fn a() -> Result<i32, ()> {
    //~^ WARN: `Result` is ambiguous
    //~| WARN: this was previously accepted by the compiler but is being phased out; it will become a hard error in a future release!
    //~| ERROR: type alias takes 1 generic argument but 2 generic arguments were supplied
    Ok(1)
}

pub fn main() {}
