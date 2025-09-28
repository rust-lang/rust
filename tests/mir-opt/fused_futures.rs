//@ edition:2024
//@ compile-flags: -Zfused-futures
// skip-filecheck
// EMIT_MIR_FOR_EACH_PANIC_STRATEGY

#![feature(coroutines, stmt_expr_attributes)]
#![allow(unused)]

// EMIT_MIR fused_futures.future-{closure#0}.coroutine_resume.0.mir
pub async fn future() -> u32 {
    42
}

// EMIT_MIR fused_futures.main-{closure#0}.coroutine_resume.0.mir
fn main() {
    let mut coroutine = #[coroutine]
    || {
        yield 1;
        return "foo";
    };
}
