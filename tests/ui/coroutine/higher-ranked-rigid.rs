//@ edition: 2024
//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver
//@ check-pass

// Regression test for <https://github.com/rust-lang/trait-system-refactor-initiative/issues/177>.
// Coroutines erase all free lifetimes from their interior types, replacing them with higher-
// ranked regions which act as universals, to properly represent the fact that we don't know what
// the value of the region is within the coroutine.
//
// In the future in `from_request`, that means that the `'r` lifetime is being replaced in
// `<T as FromRequest<'r>>::Assoc`, which is in present in the existential bounds of the
// `dyn Future` that it's awaiting. Normalizing this associated type, with its free lifetimes
// replaced, means proving `T: FromRequest<'!0>`, which doesn't hold without constraining the
// `'!0` lifetime, which we don't do today.

// Proving `T: Trait` holds when `<T as Trait>::Assoc` is rigid is not necessary for soundness,
// at least not *yet*, and it's not even necessary for diagnostics since we have other special
// casing for, e.g., AliasRelate goals failing in the BestObligation folder.

// The old solver unintentioanlly avoids this by never checking that `T: Trait` holds when
// `<T as Trait>::Assoc` is rigid. Introducing this additional requirement when projecting rigidly
// in the old solver causes this (and tons of production crates) to fail. See the fallout from the
// crater run at <https://github.com/rust-lang/rust/pull/139763>.

use std::future::Future;
use std::pin::Pin;

pub trait FromRequest<'r> {
    type Assoc;
    fn from_request() -> Pin<Box<dyn Future<Output = Self::Assoc> + Send>>;
}

fn test<'r, T: FromRequest<'r>>() -> Pin<Box<dyn Future<Output = ()> + Send>> {
    Box::pin(async move {
        T::from_request().await;
    })
}

fn main() {}
