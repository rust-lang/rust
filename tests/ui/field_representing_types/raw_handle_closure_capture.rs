//@ run-pass
//@ edition: 2024

#![expect(incomplete_features)]
#![feature(field_projections)]

use std::ops::place::{LocalHandle, raw_handle};

struct S {
    a: i32,
    b: i32,
}

fn call_fn<R, F: Fn() -> R>(f: F) -> R {
    f()
}

fn call_fn_mut<R, F: FnMut() -> R>(mut f: F) -> R {
    f()
}

fn main() {
    let s = S { a: 10, b: 20 };
    let c_imm = || unsafe { raw_handle!(s.a) };
    let h_imm: LocalHandle<i32> = call_fn(c_imm);
    assert_eq!(unsafe { *h_imm.as_ptr() }, 10);
    assert_eq!(s.b, 20);

    let mut s_mut = S { a: 1, b: 2 };
    let c_mut = || unsafe {
        *raw_handle!(s_mut.b).as_mut_ptr() = 42;
    };
    call_fn_mut(c_mut);
    assert_eq!(s_mut.a, 1);
    assert_eq!(s_mut.b, 42);

    let mut target = S { a: 7, b: 8 };
    let r = &mut target;
    let c_move = move || unsafe {
        *raw_handle!(r.a).as_mut() = 99;
    };
    call_fn_mut(c_move);
    assert_eq!(target.a, 99);
    assert_eq!(target.b, 8);
}
