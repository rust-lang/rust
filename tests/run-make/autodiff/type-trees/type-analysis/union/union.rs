#![feature(autodiff)]

use std::autodiff::autodiff_reverse;

#[allow(dead_code)]
union MyUnion {
    f: f32,
    i: i32,
}

#[autodiff_reverse(d_square, Duplicated, Active)]
#[no_mangle]
fn callee(x: &MyUnion) -> f32 {
    unsafe { x.f * x.f }
}

fn main() {
    let x = MyUnion { f: 7.0 };
    let _ = callee(&x);
}
