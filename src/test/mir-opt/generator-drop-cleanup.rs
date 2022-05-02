#![feature(generators, generator_trait)]

// needs-unwind

// Regression test for #58892, generator drop shims should not have blocks
// spuriously marked as cleanup

// EMIT_MIR generator_drop_cleanup.main-{closure#0}.generator_drop.0.mir
fn main() {
    let gen = || {
        let _s = String::new();
        yield;
    };
}
