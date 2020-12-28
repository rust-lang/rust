#![feature(no_prelude)]
#![no_prelude]

// Test that things from the prelude aren't in scope. Use many of them
// so that renaming some things won't magically make this test fail
// for the wrong reason (e.g. if `Add` changes to `Addition`, and
// `no_prelude` stops working, then the `impl Add` will still
// fail with the same error message).

struct Test;
impl Add for Test {} //~ ERROR: not in scope
impl Clone for Test {} //~ ERROR: not in scope
impl Iterator for Test {} //~ ERROR: not in scope
impl ToString for Test {} //~ ERROR: not in scope
impl Writer for Test {} //~ ERROR: not in scope

fn main() {
    drop(2) //~ ERROR: unresolved name
}
