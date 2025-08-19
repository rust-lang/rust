//@ known-bug: unknown
// Ensure that we print unsatisfied always-const trait bounds as `const Trait` in diagnostics.
//@ compile-flags: -Znext-solver

#![feature(const_trait_impl, generic_const_exprs)]
#![allow(incomplete_features)]

fn require<T: const Trait>() {}

#[const_trait]
trait Trait {
    fn make() -> u32;
}

struct Ty;

impl Trait for Ty {
    fn make() -> u32 { 0 }
}

fn main() {
    require::<Ty>();
}

struct Container<const N: u32>;

// FIXME(const_trait_impl): Somehow emit `the trait bound `T: const Trait`
// is not satisfied` here instead and suggest changing `Trait` to `const Trait`.
fn accept0<T: Trait>(_: Container<{ T::make() }>) {}

// FIXME(const_trait_impl): Instead of suggesting `+ const Trait`, suggest
//                 changing `[const] Trait` to `const Trait`.
const fn accept1<T: [const] Trait>(_: Container<{ T::make() }>) {}
