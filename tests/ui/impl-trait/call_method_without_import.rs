//! Test that opaque types only pick up methods from traits in their bounds
//! if the trait is imported.
//!
//! FIXME: always look through the bounds of an opaque type to see if there are
//! methods that could be called on any of the bound traits, irrespective of
//! imported traits.

//@ revisions: import no_import
//@[import] check-pass

#[cfg(import)]
use std::fmt::Debug as _;

fn foo(f: &mut std::fmt::Formatter<'_>) -> impl std::fmt::Debug {
    if false {
        let x = foo(f);
        x.fmt(f);
        //[no_import]~^ ERROR: no method named `fmt` found
    }
    ()
}

fn foo1(f: &mut std::fmt::Formatter<'_>) -> impl std::fmt::Debug {
    if false {
        let x = &mut foo(f);
        x.fmt(f);
        //[no_import]~^ ERROR: no method named `fmt` found
    }
    ()
}

// inconsistent with this
fn bar<T>(t: impl std::fmt::Debug, f: &mut std::fmt::Formatter<'_>) {
    t.fmt(f);
}

// and the desugared version, of course
fn baz<T: std::fmt::Debug>(t: T, f: &mut std::fmt::Formatter<'_>) {
    t.fmt(f);
}

fn main() {}
