#![feature(core_intrinsics, rustc_attrs)]

use std::intrinsics::rustc_peek;

#[rustc_mir(rustc_peek_liveness, stop_after_dataflow)]
fn foo() -> Option<i32> {
    let mut x = None;

    // `x` is live here since it is used in the next statement...
    rustc_peek(x);

    dbg!(x);

    // But not here, since it is overwritten below
    rustc_peek(x); //~ ERROR rustc_peek: bit not set

    x = Some(4);

    x
}

fn main() {}

//~? ERROR stop_after_dataflow ended compilation
