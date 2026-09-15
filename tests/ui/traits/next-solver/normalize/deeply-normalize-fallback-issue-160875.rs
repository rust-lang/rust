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
// The first normalization attempt folds the inner alias before trying the outer
// one. The outer goal then returns `NoSolution`, so `normalize_with_universes`
// discards the partial result and retries the original value with
// `ReplaceAliasWithInfer`. The nested alias stays in place because it has escaping
// bound vars, while the outer alias becomes a fresh inference variable.
//
// Fulfillment later proves the fallback obligations and constrains that variable,
// but it does not rebuild the value produced by the fold. Failing to resolve the
// value before returning made `normalize_erasing_regions` see `fn(?3t)` even though
// the inference context had already resolved it to `fn(&'?0 ())`.
//
// `bar` is a `const fn` so that drop elaboration -- and with it the
// `PostAnalysisNormalize` pass which normalizes its signature -- runs even
// though this test only emits metadata. The body error makes this test reach the
// MIR assertion; a valid body takes the same normalization fallback but does not
// reach that assertion here.

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
