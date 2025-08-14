// EMIT_MIR_FOR_EACH_PANIC_STRATEGY
// skip-filecheck
//@ test-mir-pass: Inline
//@ edition: 2021
//@ compile-flags: -Zinline-mir --crate-type=lib

// EMIT_MIR inline_default_trait_body.Trait-a.Inline.diff
// EMIT_MIR inline_default_trait_body.Trait-b.Inline.diff
pub trait Trait {
    fn a(&self) {
        ().b();
    }

    fn b(&self) {
        ().a();
    }
}

impl Trait for () {}
