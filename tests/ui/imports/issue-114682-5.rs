// edition: 2018
// aux-build: issue-114682-5-extern-1.rs
// aux-build: issue-114682-5-extern-2.rs
// compile-flags: --extern issue_114682_5_extern_1

// https://github.com/rust-lang/rust/pull/114682#issuecomment-1880755441
extern crate issue_114682_5_extern_2;

use issue_114682_5_extern_2::p::*;
use issue_114682_5_extern_1::Url;
//~^ WARN: `issue_114682_5_extern_1` is ambiguous
//~| WARN: this was previously accepted by the compiler but is being phased out; it will become a hard error in a future release!
//~| ERROR: unresolved import `issue_114682_5_extern_1::Url`
//~| ERROR: `issue_114682_5_extern_1` is ambiguous

fn main() {}
