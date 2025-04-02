// skip-filecheck
#![feature(coroutines, coroutine_trait, stmt_expr_attributes)]

// EMIT_MIR_FOR_EACH_PANIC_STRATEGY

// Regression test for #58892, coroutine drop shims should not have blocks
// spuriously marked as cleanup

// EMIT_MIR coroutine_drop_cleanup.main-{closure#0}.coroutine_drop.0.mir
fn main() {
    let gen_ = #[coroutine]
    || {
        let _s = String::new();
        yield;
    };
}
