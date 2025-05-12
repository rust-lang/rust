// General test of maybe_uninits state computed by MIR dataflow.

#![feature(core_intrinsics, rustc_attrs)]

use std::intrinsics::rustc_peek;
use std::mem::{drop, replace};

struct S(i32);

#[rustc_mir(rustc_peek_maybe_uninit,stop_after_dataflow)]
fn foo(x: &mut S) {
    // `x` is initialized here, so maybe-uninit bit is 0.

    rustc_peek(&x); //~ ERROR rustc_peek: bit not set

    ::std::mem::drop(x);

    // `x` definitely uninitialized here, so maybe-uninit bit is 1.
    rustc_peek(&x);
}
fn main() {
    foo(&mut S(13));
    foo(&mut S(13));
}

//~? ERROR stop_after_dataflow ended compilation
