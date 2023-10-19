// skip-filecheck
#![feature(coroutines, coroutine_trait)]

// EMIT_MIR_FOR_EACH_PANIC_STRATEGY

// Regression test for #58892, coroutine drop shims should not have blocks
// spuriously marked as cleanup

// EMIT_MIR coroutine_drop_cleanup.main-{closure#0}.coroutine_drop.0.mir
fn main() {
    let gen = || {
        let _s = String::new();
        yield;
    };
}
