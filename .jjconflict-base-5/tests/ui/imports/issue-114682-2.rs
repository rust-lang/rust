//@ aux-build: issue-114682-2-extern.rs
// https://github.com/rust-lang/rust/pull/114682#issuecomment-1879998900

extern crate issue_114682_2_extern;

use issue_114682_2_extern::max; //~ ERROR `max` is ambiguous
                                //~| WARN this was previously accepted

type A = issue_114682_2_extern::max; //~ ERROR `max` is ambiguous
                                     //~| WARN this was previously accepted

fn main() {}
