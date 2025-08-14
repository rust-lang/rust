//@ edition:2018
//@ aux-build:edition-lint-infer-outlives-macro.rs
//@ run-rustfix

#![deny(explicit_outlives_requirements)]
#![allow(dead_code)]

#[macro_use]
extern crate edition_lint_infer_outlives_macro;

// Test that the lint does not fire if the predicate is from the local crate,
// but all the bounds are from an external macro.
macro_rules! make_foo {
    ($a:tt) => {
        struct Foo<$a, 'b: $a> {
            foo: &$a &'b (),
        }

        struct FooWhere<$a, 'b> where 'b: $a {
            foo: &$a &'b (),
        }
    }
}

gimme_a! {make_foo!}

struct Bar<'a, 'b: 'a> {
    //~^ ERROR: outlives requirements can be inferred
    bar: &'a &'b (),
}

struct BarWhere<'a, 'b> where 'b: 'a {
    //~^ ERROR: outlives requirements can be inferred
    bar: &'a &'b (),
}

// Test that the lint *does* fire if the predicate is contained in a local macro.
mod everything_inside {
    macro_rules! m {
        ('b: 'a) => {
            struct Foo<'a, 'b: 'a>(&'a &'b ());
            //~^ ERROR: outlives requirements can be inferred
            struct Bar<'a, 'b>(&'a &'b ()) where 'b: 'a;
            //~^ ERROR: outlives requirements can be inferred
            struct Baz<'a, 'b>(&'a &'b ()) where (): Sized, 'b: 'a;
            //~^ ERROR: outlives requirements can be inferred
        };
    }
    m!('b: 'a);
}

mod inner_lifetime_outside_colon_inside {
    macro_rules! m {
        ($b:lifetime: 'a) => {
            struct Foo<'a, $b: 'a>(&'a &$b ());
            //~^ ERROR: outlives requirements can be inferred
            struct Bar<'a, $b>(&'a &$b ()) where $b: 'a;
            //~^ ERROR: outlives requirements can be inferred
            struct Baz<'a, $b>(&'a &$b ()) where (): Sized, $b: 'a;
            //~^ ERROR: outlives requirements can be inferred
        }
    }
    m!('b: 'a);
}

mod outer_lifetime_outside_colon_inside {
    macro_rules! m {
        ('b: $a:lifetime) => {
            struct Foo<$a, 'b: $a>(&$a &'b ());
            struct Bar<$a, 'b>(&$a &'b ()) where 'b: $a;
            struct Baz<$a, 'b>(&$a &'b ()) where (): Sized, 'b: $a;
        }
    }
    m!('b: 'a);
}

mod both_lifetimes_outside_colon_inside {
    macro_rules! m {
        ($b:lifetime: $a:lifetime) => {
            struct Foo<$a, $b: $a>(&$a &$b ());
            struct Bar<$a, $b>(&$a &$b ()) where $b: $a;
            struct Baz<$a, $b>(&$a &$b ()) where (): Sized, $b: $a;
        }
    }
    m!('b: 'a);
}

mod everything_outside {
    macro_rules! m {
        ($b:lifetime $colon:tt $a:lifetime) => {
            struct Foo<$a, $b $colon $a>(&$a &$b ());
            struct Bar<$a, $b>(&$a &$b ()) where $b $colon $a;
            struct Baz<$a, $b>(&$a &$b ()) where (): Sized, $b $colon $a;
        }
    }
    m!('b: 'a);
}

mod everything_outside_with_tt_inner {
    macro_rules! m {
        ($b:tt $colon:tt $a:lifetime) => {
            struct Foo<$a, $b $colon $a>(&$a &$b ());
            struct Bar<$a, $b>(&$a &$b ()) where $b $colon $a;
            struct Baz<$a, $b>(&$a &$b ()) where (): Sized, $b $colon $a;
        }
    }
    m!('b: 'a);
}

// FIXME: These should be consistent.
mod everything_outside_with_tt_outer {
    macro_rules! m {
        ($b:lifetime $colon:tt $a:tt) => {
            // FIXME: replacement span is corrupted due to a collision in metavar span table.
            // struct Foo<$a, $b $colon $a>(&$a &$b ());
            // ^ ERROR: outlives requirements can be inferred
            struct Bar<$a, $b>(&$a &$b ()) where $b $colon $a;
            //~^ ERROR: outlives requirements can be inferred
            struct Baz<$a, $b>(&$a &$b ()) where (): Sized, $b $colon $a;
            //~^ ERROR: outlives requirements can be inferred
        }
    }
    m!('b: 'a);
}

mod everything_outside_with_tt_both {
    macro_rules! m {
        ($b:tt $colon:tt $a:tt) => {
            // FIXME: replacement span is corrupted due to a collision in metavar span table.
            // struct Foo<$a, $b $colon $a>(&$a &$b ());
            // ^ ERROR: outlives requirements can be inferred
            struct Bar<$a, $b>(&$a &$b ()) where $b $colon $a;
            //~^ ERROR: outlives requirements can be inferred
            struct Baz<$a, $b>(&$a &$b ()) where (): Sized, $b $colon $a;
            //~^ ERROR: outlives requirements can be inferred
        }
    }
    m!('b: 'a);
}

fn main() {}
