// General test of maybe_uninits state computed by MIR dataflow.

#![feature(core_intrinsics, rustc_attrs)]

use std::intrinsics::rustc_peek;
use std::mem::{drop, replace};

struct S(i32);

#[rustc_mir(rustc_peek_definite_init,stop_after_dataflow)]
fn foo(test: bool, x: &mut S, y: S, mut z: S) -> S {
    let ret;
    // `ret` starts off uninitialized
    unsafe { rustc_peek(&ret); }  //~ ERROR rustc_peek: bit not set

    // All function formal parameters start off initialized.

    unsafe { rustc_peek(&x) };
    unsafe { rustc_peek(&y) };
    unsafe { rustc_peek(&z) };

    ret = if test {
        ::std::mem::replace(x, y)
    } else {
        z = y;
        z
    };

    // `z` may be uninitialized here.
    unsafe { rustc_peek(&z); } //~ ERROR rustc_peek: bit not set

    // `y` is definitely uninitialized here.
    unsafe { rustc_peek(&y); } //~ ERROR rustc_peek: bit not set

    // `x` is still (definitely) initialized (replace above is a reborrow).
    unsafe { rustc_peek(&x); }

    ::std::mem::drop(x);

    // `x` is *definitely* uninitialized here
    unsafe { rustc_peek(&x); } //~ ERROR rustc_peek: bit not set

    // `ret` is now definitely initialized (via `if` above).
    unsafe { rustc_peek(&ret); }

    ret
}
fn main() {
    foo(true, &mut S(13), S(14), S(15));
    foo(false, &mut S(13), S(14), S(15));
}
