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

#[autodiff_reverse(d_square, Duplicated, Active)]
#[no_mangle]
fn callee(x: &[f32; 3]) -> f32 {
    x[0] * x[0] + x[1] * x[1] + x[2] * x[2]
}

fn main() {
    let x = [1.0f32, 2.0, 3.0];
    let mut df_dx = [0.0f32; 3];
    let out = callee(&x);
    let out_ = d_square(&x, &mut df_dx, 1.0);
    assert_eq!(out, out_);
    assert_eq!(2.0, df_dx[0]);
    assert_eq!(4.0, df_dx[1]);
    assert_eq!(6.0, df_dx[2]);
}
