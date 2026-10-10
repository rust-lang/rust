#![feature(core_intrinsics, rustc_attrs)]

use std::intrinsics::rustc_peek;

#[rustc_mir(rustc_peek_liveness, stop_after_dataflow)]
fn foo() -> i32 {
    let mut x: i32;
    let mut p: *const i32;

    x = 0;

    // `x` is live here since it is used in the next statement...
    rustc_peek(x);

    p = &x;

    // ... but not here, even while it can be accessed through `p`.
    rustc_peek(x); //~ ERROR rustc_peek: bit not set
    let tmp = unsafe { *p };

    x = tmp + 1;

    rustc_peek(x);

    x
}

fn main() {}

//~? ERROR stop_after_dataflow ended compilation
