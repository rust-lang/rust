// Regression test for #157854.
//
// An ICE ("Found unstable fingerprints for evaluate_obligation") occurred during
// incremental compilation when:
// 1. The first compilation succeeds, caching `evaluate_obligation` results.
// 2. A subsequent compilation introduces an error, causing the inference context
//    to become "tainted by errors".
// 3. The tainted-by-errors path in trait evaluation cached stack-dependent
//    (cycle-participant) results into the *global* evaluation cache.
// 4. During incremental verification, the fingerprint of the globally cached
//    result didn't match the recomputed one, triggering the ICH assertion.
//
// The fix ensures that when caching under the tainted-by-errors path, only the
// local (per-InferCtxt) cache is used, not the global (per-TyCtxt) cache that
// feeds into incremental compilation.

//@ revisions: rpass1 bfail2

struct S(&'static Box<S>);

trait I<T> {}
impl<F: Fn()> I<()> for F {}
impl<F: Fn(T), T: Sync> I<(T,)> for F {}

fn f<T>(_: impl I<T>) {}

fn main() {
    #[cfg(rpass1)]
    f(|_: S| {});

    #[cfg(bfail2)]
    f(|_: S| { foo; });
    //[bfail2]~^ ERROR cannot find value `foo` in this scope
}
