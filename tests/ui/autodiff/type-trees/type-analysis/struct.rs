//@ run-pass
//@ compile-flags: -Zautodiff=Enable,PrintTAFn=callee -Zautodiff=NoPostopt -C opt-level=3 -Clto=fat -g
//@ no-prefer-dynamic
//@ needs-enzyme
//@ normalize-stderr: "!(dbg|noundef) ![0-9]+" -> "!$1 !N"
//@ normalize-stderr: "%[0-9]+" -> "%X"
//@ normalize-stdout: "!(dbg|noundef) ![0-9]+" -> "!$1 !N"
//@ normalize-stdout: "%[0-9]+" -> "%X"

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
