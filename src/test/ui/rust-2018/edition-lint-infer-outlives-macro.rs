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

        struct Foo2<$a, 'b: $a> {
            foo: &$a &'b (),
        }
    };
}

gimme_a! {make_foo!}

struct Bar<'a, 'b: 'a> {
    //~^ ERROR: outlives requirements can be inferred
    bar: &'a &'b (),
}

macro_rules! make_quux {
    () => {
        struct Quux<'a, 'b> where 'b: 'a {
            //~^ ERROR: outlives requirements can be inferred
            baz: &'a &'b (),
        }

        struct Quux2<'a, 'b: 'a> {
            //~^ ERROR: outlives requirements can be inferred
            baz: &'a &'b (),
        }
    };
}

make_quux!{}

macro_rules! make_baz {
    () => {
        make_baz!{ 'a }
    };
    ($a:lifetime) => {
        struct Baz<$a, 'b> where 'b: $a {
            baz: &$a &'b (),
        }

        struct Baz2<$a, 'b: $a> {
            baz: &$a &'b (),
        }
    };
}

make_baz!{ 'a }

mod baz {
    make_baz!{}
}

fn main() {}
