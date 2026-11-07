//@ edition:2018

// Tests that arbitrary crates (other than `core`, `std` and `meta`)
// aren't allowed without `--extern`, even if they're in the sysroot.
use alloc; //~ ERROR unresolved import `alloc`
use test; // OK, imports the built-in attribute macro `#[test]`, but not the `test` crate.
use proc_macro; // OK, imports the built-in `proc_macro` attribute, but not the `proc_macro` crate.

fn main() {}
