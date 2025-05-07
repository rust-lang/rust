// Support for legacy derive helpers is limited and heuristic-based
// (that's exactly the reason why they are deprecated).

//@ edition:2018
//@ proc-macro: test-macros.rs

#[macro_use]
extern crate test_macros;

use derive as my_derive;

#[my_derive(Empty)]
#[empty_helper] // OK
struct S1;

// Legacy helper detection doesn't see through `derive` renaming.
#[empty_helper] //~ ERROR cannot find attribute `empty_helper` in this scope
#[my_derive(Empty)]
struct S2;

fn main() {}
