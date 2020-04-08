#![feature(generators, generator_trait)]

// Regression test for #58892, generator drop shims should not have blocks
// spuriously marked as cleanup

// EMIT_MIR rustc.main-{{closure}}.generator_drop.0.mir
fn main() {
    let gen = || {
        yield;
    };
}
