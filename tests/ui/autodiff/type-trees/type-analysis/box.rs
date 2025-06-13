//@ run-pass
//@ compile-flags: -Zautodiff=Enable,PrintTAFn=callee -Zautodiff=NoPostopt -C opt-level=3 -Clto=fat -g
//@ no-prefer-dynamic
//@ needs-enzyme
//@ normalize-stderr: "!(dbg|noundef) ![0-9]+" -> "!$1 !N"
//@ normalize-stderr: "%[0-9]+" -> "%X"
//@ normalize-stderr: "!nonnull ![0-9]+" -> "!nonnull !N"
//@ normalize-stderr: "!align ![0-9]+" -> "!align !N"
//@ normalize-stdout: "!(dbg|noundef) ![0-9]+" -> "!$1 !N"
//@ normalize-stdout: "%[0-9]+" -> "%X"
//@ normalize-stdout: "!nonnull ![0-9]+" -> "!nonnull !N"
//@ normalize-stdout: "!align ![0-9]+" -> "!align !N"

#![feature(autodiff)]

use std::autodiff::autodiff_reverse;

#[autodiff_reverse(d_square, Duplicated, Active)]
#[no_mangle]
fn callee(x: &Box<f32>) -> f32 {
    **x * **x
}

fn main() {
    let x = Box::new(7.0f32);
    let mut df_dx = Box::new(0.0f32);
    let out = callee(&x);
    let out_ = d_square(&x, &mut df_dx, 1.0);
    assert_eq!(out, out_);
    assert_eq!(14.0, *df_dx);
}
