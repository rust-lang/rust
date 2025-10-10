// skip-filecheck
//@ edition:2024
//@ compile-flags: -Zmir-opt-level=0 -C panic=abort

#![feature(stmt_expr_attributes)]
#![feature(closure_track_caller)]
#![feature(coroutine_trait)]
#![feature(coroutines)]

use std::ops::{Coroutine, CoroutineState};
use std::panic::Location;
use std::pin::Pin;

// EMIT_MIR coroutine.main-{closure#0}.StateTransform.after.mir
// EMIT_MIR coroutine.main-{closure#1}.StateTransform.after.mir
fn main() {
    let simple = #[coroutine]
    |arg: String| {
        yield ("first", arg.clone(), Location::caller());
        yield ("second", arg.clone(), Location::caller());
    };

    let track_caller = #[track_caller]
    #[coroutine]
    |arg: String| {
        yield ("first", arg.clone(), Location::caller());
        yield ("second", arg.clone(), Location::caller());
    };
}
