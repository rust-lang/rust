//@ revisions: old next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver
//@[next] dont-require-annotations: ERROR
//@[old] failure-status: 101
//@[old] dont-check-compiler-stderr
//@[old] known-bug: #148095

// A regression test for #148095. The old solver incorrectly cached
// a result from in one context and then used that result in a different
// context, triggering an assert.
//
// More specifically, when in `self.query_mode == TraitQueryMode::Canonical`,
// `WfPredicates::normalize` creates a separate `SelectionContext` for
// normalization which is not in that mode. That then ICEs when accessing
// a cache entry from the `infcx`-local cache, which also caches canonical
// overflow.
//
// This test results in a lot of errors with the new solver, so lets not add
// explicit annotations here.

use std::ops::Mul;

struct Quantity<S>(S);
impl<S> Mul<Quantity<<f32 as Mul<S>>::Output>> for f32
where
    Quantity<Self::Output>:
{
    type Output = ();
    fn mul(self, _: Quantity<<f32 as Mul<S>>::Output>) {}
}

fn main() {}
