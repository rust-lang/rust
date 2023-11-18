// check-pass
// aux-build: issue-114682-6-extern.rs

// https://github.com/rust-lang/rust/pull/114682#issuecomment-1880755441
extern crate issue_114682_6_extern;

use issue_114682_6_extern::*;

fn main() {
    let log = 2;
    //~^ WARN: `log` is ambiguous
    //~| WARN: this was previously accepted by the compiler but is being phased out; it will become a hard error in a future release!
    let _ = log;
}
