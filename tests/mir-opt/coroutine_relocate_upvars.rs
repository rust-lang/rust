// skip-filecheck
//@ compile-flags: -Zpack-coroutine-layout=captures-only
#![feature(coroutines, coroutine_trait, stmt_expr_attributes)]

// EMIT_MIR_FOR_EACH_PANIC_STRATEGY

// EMIT_MIR coroutine_relocate_upvars.main-{closure#0}.RelocateUpvars.before.mir
// EMIT_MIR coroutine_relocate_upvars.main-{closure#0}.RelocateUpvars.after.mir
fn main() {
    let mut x = String::new();
    let gen_ = #[coroutine]
    || {
        x = String::new();
        yield;
    };
}
