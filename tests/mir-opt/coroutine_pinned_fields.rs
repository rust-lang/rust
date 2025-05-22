// skip-filecheck

//! Ensures pinned coroutine fields are marked correctly

//@ compile-flags: -C panic=abort
//@ edition: 2024

#![crate_type = "lib"]
#![feature(coroutines, coroutine_trait, stmt_expr_attributes)]

use std::future::Future;
use std::ops::Coroutine;

fn do_thing<T>(_: T) {}

// EMIT_MIR coroutine_pinned_fields.use_borrow_across_yield-{closure#0}.coroutine_pre-elab.0.mir
fn use_borrow_across_yield() -> impl Coroutine {
    #[coroutine]
    static || {
        let mut a = 19; // pinned
        let b = &mut a; // not pinned
        yield;
        *b = 23;
        yield;
        a
    }
}

// EMIT_MIR coroutine_pinned_fields.borrow_not_held_across_yield-{closure#0}.coroutine_pre-elab.0.mir
fn borrow_not_held_across_yield() -> impl Coroutine {
    #[coroutine]
    static || {
        // NOTE: unfortunately, this field is currently marked as pinned even though it shouldn't be
        let mut x = 9; // not pinned
        {
            let y = &mut x; // not stored
            *y += 5;
        }
        yield;
        x
    }
}

async fn nop() {}

// EMIT_MIR coroutine_pinned_fields.async_block-{closure#0}.coroutine_pre-elab.0.mir
fn async_block() -> impl Future {
    async {
        let mut x = 9; // pinned
        let y = &mut x; // not pinned
        nop().await;
        *y += 1;
        nop().await;
        x
    }
}

// EMIT_MIR coroutine_pinned_fields.async_fn_borrow_not_used_after_yield-{closure#0}.coroutine_pre-elab.0.mir
async fn async_fn_borrow_not_used_after_yield() {
    let string = String::from("abc123");
    let str = string.as_str();
    do_thing(str);

    nop().await;
}

// EMIT_MIR coroutine_pinned_fields.movable_is_never_pinned-{closure#0}.coroutine_pre-elab.0.mir
fn movable_is_never_pinned() -> impl Coroutine {
    #[coroutine]
    || {
        let mut bar = 29;
        yield;
        bar += 2;
        yield;
        bar
    }
}
