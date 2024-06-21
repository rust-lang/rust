#![no_implicit_prelude]

// Test that things from the prelude aren't in scope. Use many of them
// so that renaming some things won't magically make this test fail
// for the wrong reason (e.g., if `Add` changes to `Addition`, and
// `no_implicit_prelude` stops working, then the `impl Add` will still
// fail with the same error message).

struct Test;
impl Add for Test {} //~ ERROR cannot find trait `Add`
impl Clone for Test {} //~ ERROR expected trait, found derive macro `Clone`
impl Iterator for Test {} //~ ERROR cannot find trait `Iterator`
impl ToString for Test {} //~ ERROR cannot find trait `ToString`
impl Writer for Test {} //~ ERROR cannot find trait `Writer`

fn main() {
    drop(2) //~ ERROR cannot find function `drop`
}
