//@ edition: 2018
//@ aux-build: issue-114682-5-extern-1.rs
//@ aux-build: issue-114682-5-extern-2.rs
//@ compile-flags: --extern issue_114682_5_extern_1
// https://github.com/rust-lang/rust/pull/114682#issuecomment-1880755441

extern crate issue_114682_5_extern_2;

use issue_114682_5_extern_2::p::*;
use issue_114682_5_extern_1::Url;
//~^ ERROR `issue_114682_5_extern_1` is ambiguous
//~| ERROR `issue_114682_5_extern_1` is ambiguous
//~| ERROR unresolved import `issue_114682_5_extern_1::Url`
//~| WARN this was previously accepted

fn main() {}
