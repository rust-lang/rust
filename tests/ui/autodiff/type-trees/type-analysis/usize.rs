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
fn callee(x: &usize) -> usize {
    *x * *x
}

fn main() {
    let x: usize = 7;
    let _ = callee(&x);
}
