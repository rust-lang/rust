// skip-filecheck
//@ compile-flags: -Zpack-coroutine-layout=captures-only
#![feature(coroutines, coroutine_trait, stmt_expr_attributes)]

// EMIT_MIR_FOR_EACH_PANIC_STRATEGY

// EMIT_MIR coroutine_relocate_upvars.main-{closure#0}.RelocateUpvars.diff
// EMIT_MIR coroutine_relocate_upvars.main-{closure#1}.RelocateUpvars.diff
fn main() {
    let mut x = String::new();
    let capture_by_ref = #[coroutine]
    || {
        x = String::new();
        yield;
    };
    let mut y = String::new();
    let capture_by_val = #[coroutine]
    || {
        y = String::new();
        yield;
    };
}
