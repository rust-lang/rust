//@ compile-flags: -Zautodiff=Enable,NoTT,NoPostopt -C no-prepopulate-passes -C opt-level=3  -Clto=fat
//@ no-prefer-dynamic
//@ needs-enzyme

#![feature(autodiff)]
use std::autodiff::*;

// Usually we would store the return value of the differentiated function.
// However, if the return type is void or an empty struct,
// we don't need to store anything. Verify this, since it caused a bug.

// CHECK:; void_ret::main
// CHECK-NEXT:    ; Function Attrs:
// CHECK-NEXT:    define internal
// CHECK-NOT: store {} undef, ptr undef
// CHECK: ret void

#[autodiff_reverse(bar, Duplicated, Duplicated)]
pub fn foo(r: &[f64; 10], res: &mut f64) {
    let mut output = [0.0; 10];
    output[0] = r[0];
    output[1] = r[1] * r[2];
    output[2] = r[4] * r[5];
    output[3] = r[2] * r[6];
    output[4] = r[1] * r[7];
    output[5] = r[2] * r[8];
    output[6] = r[1] * r[9];
    output[7] = r[5] * r[6];
    output[8] = r[5] * r[7];
    output[9] = r[4] * r[8];
    *res = output.iter().sum();
}
fn main() {
    let inputs = Box::new([3.1; 10]);
    let mut d_inputs = Box::new([0.0; 10]);
    let mut res = Box::new(0.0);
    let mut d_res = Box::new(1.0);

    bar(&inputs, &mut d_inputs, &mut res, &mut d_res);
    dbg!(&d_inputs);
}
