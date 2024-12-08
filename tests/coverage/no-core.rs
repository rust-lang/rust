#![feature(no_core)]
#![no_core]
//@ edition: 2021

// Test that coverage instrumentation works for `#![no_core]` crates.

// For this test, we pull in std anyway, to avoid having to set up our own
// no-core or no-std environment. What's important is that the compiler allows
// coverage for a crate with the `#![no_core]` annotation.
extern crate std;

fn main() {}
