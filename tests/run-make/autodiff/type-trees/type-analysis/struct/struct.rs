#![feature(autodiff)]

use std::autodiff::autodiff_reverse;

#[derive(Copy, Clone)]
struct MyStruct {
    f: f32,
}

#[autodiff_reverse(d_square, Duplicated, Active)]
#[no_mangle]
fn callee(x: &MyStruct) -> f32 {
    x.f * x.f
}

fn main() {
    let x = MyStruct { f: 7.0 };
    let mut df_dx = MyStruct { f: 0.0 };
    let out = callee(&x);
    let out_ = d_square(&x, &mut df_dx, 1.0);
    assert_eq!(out, out_);
    assert_eq!(14.0, df_dx.f);
}
