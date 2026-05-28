// General test of maybe_uninits state computed by MIR dataflow.

#![feature(core_intrinsics, rustc_attrs)]

use std::intrinsics::rustc_peek;
use std::mem::{drop, replace};

struct S(i32);

#[rustc_mir(rustc_peek_maybe_uninit,stop_after_dataflow)]
fn foo(test: bool, x: &mut S, y: S, mut z: S) -> S {
    let ret;
    // `ret` starts off uninitialized
    rustc_peek(&ret);

    // All function formal parameters start off initialized.

    rustc_peek(&x); //~ ERROR rustc_peek: bit not set
    rustc_peek(&y); //~ ERROR rustc_peek: bit not set
    rustc_peek(&z); //~ ERROR rustc_peek: bit not set

    ret = if test {
        ::std::mem::replace(x, y)
    } else {
        z = y;
        z
    };

    // `z` may be uninitialized here.
    rustc_peek(&z);

    // `y` is definitely uninitialized here.
    rustc_peek(&y);

    // `x` is still (definitely) initialized (replace above is a reborrow).
    rustc_peek(&x); //~ ERROR rustc_peek: bit not set

    ::std::mem::drop(x);

    // `x` is *definitely* uninitialized here
    rustc_peek(&x);

    // `ret` is now definitely initialized (via `if` above).
    rustc_peek(&ret); //~ ERROR rustc_peek: bit not set

    ret
}
fn main() {
    foo(true, &mut S(13), S(14), S(15));
    foo(false, &mut S(13), S(14), S(15));
}

//~? ERROR stop_after_dataflow ended compilation
