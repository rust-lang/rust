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

#[autodiff_reverse(d_callee, Duplicated, Active)]
#[no_mangle]
fn callee(x: &f64) -> f64 {
    x * x
}

fn main() {
    let x = std::hint::black_box(3.0);

    let output = callee(&x);
    assert_eq!(9.0, output);

    let mut df_dx = 0.0;
    let output_ = d_callee(&x, &mut df_dx, 1.0);
    assert_eq!(output, output_);
    assert_eq!(6.0, df_dx);
}
