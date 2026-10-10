// Test that we can propagate `T: 'a` obligations to our caller.  See
// `correct_region` for an explanation of how this test is setup; it's
// somewhat intricate.

//@ compile-flags:-Zverbose-internals

#![allow(warnings)]
#![feature(rustc_attrs)]

use std::cell::Cell;

fn with_signature<'a, T, F>(a: Cell<&'a ()>, b: T, op: F)
where
    F: FnOnce(Cell<&'a ()>, T),
{
    op(a, b)
}

fn require<'a, T>(_a: &Cell<&'a ()>, _b: &T)
where
    T: 'a,
{
}

#[rustc_regions]
fn no_region<'a, T>(a: Cell<&'a ()>, b: T) {
    with_signature(a, b, |x, y| {
        // See `correct_region`, which explains the point of this
        // test.  The only difference is that, in the case of this
        // function, there is no where clause *anywhere*, and hence we
        // get an error (but reported by the closure creator).
        require(&x, &y)
        //~^ ERROR the parameter type `T` may not live long enough
    })
}

#[rustc_regions]
fn correct_region<'a, T>(a: Cell<&'a ()>, b: T)
where
    T: 'a,
{
    with_signature(a, b, |x, y| {
        // Key point of this test:
        //
        // The *closure* is being type-checked with all of its free
        // regions "universalized". In particular, it does not know
        // that `x` has the type `Cell<&'a ()>`, but rather treats it
        // as if the type of `x` is `Cell<&'A ()>`, where `'A` is some
        // fresh, independent region distinct from the `'a` which
        // appears in the environment. The call to `require` here
        // forces us then to prove that `T: 'A`, but the closure
        // cannot do it on its own. It has to surface this requirement
        // to its creator (which knows that `'a == 'A`).
        require(&x, &y)
    })
}

#[rustc_regions]
fn wrong_region<'a, 'b, T>(a: Cell<&'a ()>, b: T)
where
    T: 'b,
{
    with_signature(a, b, |x, y| {
        // See `correct_region`
        require(&x, &y)
        //~^ ERROR the parameter type `T` may not live long enough
    })
}

#[rustc_regions]
fn outlives_region<'a, 'b, T>(a: Cell<&'a ()>, b: T)
where
    T: 'b,
    'b: 'a,
{
    with_signature(a, b, |x, y| {
        // See `correct_region`
        require(&x, &y)
    })
}

fn main() {}
