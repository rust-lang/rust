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
fn callee(arg: &std::vec::Vec<f32>) -> f32 {
    arg.iter().sum()
}

fn main() {
    let v = vec![1.0f32, 2.0, 3.0];
    let _ = callee(&v);
}
