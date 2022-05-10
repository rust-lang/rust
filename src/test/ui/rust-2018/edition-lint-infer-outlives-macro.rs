// edition:2018
// aux-build:edition-lint-infer-outlives-macro.rs

// Test that the lint does not fire if the where predicate
// is from the local crate, but all the bounds are from an
// external macro.

#![deny(explicit_outlives_requirements)]

#[macro_use]
extern crate edition_lint_infer_outlives_macro;

macro_rules! make_foo {
    ($a:tt) => {
        struct Foo<$a, 'b> where 'b: $a {
            foo: &$a &'b (),
        }
    }
}

gimme_a! {make_foo!}

struct Bar<'a, 'b: 'a> {
    //~^ ERROR: outlives requirements can be inferred
    bar: &'a &'b (),
}

fn main() {}
