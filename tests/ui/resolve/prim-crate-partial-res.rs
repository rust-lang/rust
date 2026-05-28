//@ aux-build: empty.rs

extern crate empty as usize;

fn foo() -> usize<()> { 0 }
//~^ ERROR type arguments are not allowed on builtin type `usize`

fn main() {}
