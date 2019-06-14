//! Test for Clippy lint renames.

// allow the new lint name here, to test if the new name works
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::new_without_default)]
#![allow(clippy::cognitive_complexity)]
#![allow(clippy::redundant_static_lifetimes)]
// warn for the old lint name here, to test if the renaming worked
#![warn(clippy::cyclomatic_complexity)]

#[warn(clippy::stutter)]
fn main() {}

#[warn(clippy::new_without_default_derive)]
struct Foo;

#[warn(clippy::const_static_lifetime)]
static Bar: &'static str = "baz";

impl Foo {
    fn new() -> Self {
        Foo
    }
}
