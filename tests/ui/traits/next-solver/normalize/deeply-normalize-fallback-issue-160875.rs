//@ needs-rustc-debug-assertions
//@ compile-flags: -Znext-solver

// Regression test for #160875.
//
// Eagerly normalizing the inner alias makes the outer alias fail the leak check:
// `Foo<for<'b> fn(&'b ())>` cannot be equated with the impl's `Foo<fn(&'a ())>`.
// Deep normalization must propagate that failure. Retrying the original aliases
// with `ReplaceAliasWithInfer` instead moved the lifetime equalities into nested
// goals, whose constraints were invisible to the parent's leak check.
//
// The fallback incorrectly succeeded and returned `fn(?3t)`. Resolving that value
// only hid the failed leak check. With debug assertions enabled, returning the
// unresolved value also triggered an assertion in `normalize_erasing_regions`.
//
// The invalid const body reaches MIR signature normalization in metadata-only
// tests. Keep debug assertions enabled to catch the original ICE as well.

#![feature(inherent_associated_types)]
#![allow(incomplete_features)]

struct Foo<T>(T);

impl<'a> Foo<fn(&'a ())> {
    type Assoc = &'a ();
}

const fn bar(_: fn(Foo<for<'b> fn(Foo<fn(&'b ())>::Assoc)>::Assoc)) {
    //~^ ERROR type mismatch resolving
    //~| ERROR type mismatch resolving
    x
    //~^ ERROR cannot find value `x` in this scope
}

fn main() {}
