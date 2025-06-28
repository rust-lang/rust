// Regression test for <https://github.com/rust-lang/rust/issues/133526>.

// Ensures we don't ICE when we encounter a `HostEffectPredicate` when computing
// the "item super predicates" for `Assoc`.

//@ compile-flags: -Znext-solver
//@ check-pass

#![feature(const_trait_impl)]

#[const_trait]
trait Trait {
    type Assoc: const Trait;
}

const fn needs_trait<T: [const] Trait>() {}

fn test<T: Trait>() {
    const { needs_trait::<T::Assoc>() };
}

fn main() {}
