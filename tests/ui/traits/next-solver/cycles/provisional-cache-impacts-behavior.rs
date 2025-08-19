//@ compile-flags: -Znext-solver
//@ check-pass
#![feature(rustc_attrs)]
#![rustc_no_implicit_bounds]

// A test showcasing that using a provisional cache can differ
// from only tracking stack entries.
//
// Without a provisional cache, we have the following proof tree:
//
// - (): A
//   - (): B
//     - (): A (coinductive cycle)
//     - (): C
//       - (): B (coinductive cycle)
//   - (): C
//     - (): B
//        - (): A (coinductive cycle)
//        - (): C (coinductive cycle)
//
// While with the current provisional cache implementation we get:
//
// - (): A
//   - (): B
//     - (): A (coinductive cycle)
//     - (): C
//       - (): B (coinductive cycle)
//   - (): C
//     - (): B (provisional cache hit)
//
// Note that if even if we were to expand the provisional cache hit,
// the proof tree would still be different:
//
// - (): A
//   - (): B
//     - (): A (coinductive cycle)
//     - (): C
//       - (): B (coinductive cycle)
//   - (): C
//     - (): B (provisional cache hit, expanded)
//       - (): A (coinductive cycle)
//       - (): C
//         - (): B (coinductive cycle)
//
// Theoretically, this can result in observable behavior differences
// due to incompleteness. However, this would require a very convoluted
// example and would still be sound. The difference is determinstic
// and can not be observed outside of the cycle itself as we don't move
// non-root cycle participants into the global cache.
//
// For an example of how incompleteness could impact the observable behavior here, see
//
//   tests/ui/traits/next-solver/cycles/coinduction/incompleteness-unstable-result.rs
#[rustc_coinductive]
trait A {}

#[rustc_coinductive]
trait B {}

#[rustc_coinductive]
trait C {}

impl<T: B + C> A for T {}
impl<T: A + C> B for T {}
impl<T: B> C for T {}

fn impls_a<T: A>() {}

fn main() {
    impls_a::<()>();
}
