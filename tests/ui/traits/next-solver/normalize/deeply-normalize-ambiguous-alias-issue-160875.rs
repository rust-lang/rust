//@ needs-rustc-debug-assertions
//@ compile-flags: -Znext-solver

// Regression test for #160875.
//
// This needs a rustc with debug assertions: the only symptom of the bug is the
// `debug_assert_eq!` in `try_normalize_after_erasing_regions`. That function
// resolves inference variables itself before returning, so without the
// assertion the pre-fix and post-fix output are identical and there is nothing
// for a test to observe.
//
// Deep normalization replaces an alias whose `NormalizesTo` goal is ambiguous
// with a fresh infer var and defers that goal. Proving the deferred goal is what
// constrains the var, but the folded value still contains its original inference
// node. Failing to resolve that stale node made `normalize_erasing_regions` trip
// its `debug_assert_eq!`.
//
// `bar` is a `const fn` so that drop elaboration -- and with it the
// `PostAnalysisNormalize` pass which normalizes its signature -- runs even
// though this test only emits metadata. Its body has to fail type check for the
// normalization of the outer alias to stay ambiguous.

#![feature(inherent_associated_types)]
#![allow(incomplete_features)]

struct Foo<T>(T);

impl<'a> Foo<fn(&'a ())> {
    type Assoc = &'a ();
}

const fn bar(_: fn(Foo<for<'b> fn(Foo<fn(&'b ())>::Assoc)>::Assoc)) {
    //~^ ERROR lifetime bound not satisfied
    //~| ERROR lifetime bound not satisfied
    x
    //~^ ERROR cannot find value `x` in this scope
}

fn main() {}
