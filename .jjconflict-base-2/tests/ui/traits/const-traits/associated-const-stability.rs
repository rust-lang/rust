//@ check-pass
//@ aux-build: associated-const-stability.rs

extern crate associated_const_stability;

use associated_const_stability::{Probe, TraitWithConstGate};

// Demonstrate that it's possible to use a `const` associated item in a stable `const` context
// even when the item is defined inside a const-unstable enclosing parent item.
//
// Here `TraitWithConstGate` is const-unstable, as is its `const impl` for `Probe`.
// This crate *does not* enable the `const_trait_gate` feature for those const impls.
// By showing that this crate compiles without that feature, while having const contexts
// that use both `<Probe as TraitWithConstGate>::ASSOC` specifically
// and generic `<T as TraitWithConstGate>::ASSOC`, we prove that the associated consts respect
// the item-level stability like `#[stable]`, not const-stability like `#[rustc_const_unstable]`.

const VALUE: usize = <Probe as TraitWithConstGate>::ASSOC;

const fn example<T: TraitWithConstGate>(_: &T) -> usize {
    <T as TraitWithConstGate>::ASSOC
}

fn main() {}
