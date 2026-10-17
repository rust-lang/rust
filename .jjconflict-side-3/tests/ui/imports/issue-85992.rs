//@ edition: 2021
//@ compile-flags: --extern issue_85992_extern --extern empty
//@ aux-build: issue-85992-extern.rs
//@ aux-build: empty.rs

issue_85992_extern::m!();

use crate::empty;
//~^ ERROR unresolved import `crate::empty`

fn main() {}
