#![feature(generators, generator_trait)]

// ignore-wasm32-bare compiled with panic=abort by default

// Regression test for #58892, generator drop shims should not have blocks
// spuriously marked as cleanup

// EMIT_MIR generator_drop_cleanup.main-{{closure}}.generator_drop.0.mir
fn main() {
    let gen = || {
        let _s = String::new();
        yield;
    };
}
