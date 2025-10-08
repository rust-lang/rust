//@ revisions: DEBUG RELEASE
//@[RELEASE] compile-flags: -Zautodiff=Enable,NoTT -C opt-level=3 -Clto=fat
//@[DEBUG]   compile-flags: -Zautodiff=Enable,NoTT -C opt-level=0 -Clto=fat -C debuginfo=2
//@ needs-enzyme
//@ incremental
//@ no-prefer-dynamic
//@ build-pass
#![crate_type = "bin"]
#![feature(autodiff)]

// We used to use llvm's metadata to instruct enzyme how to differentiate a function.
// In debug mode we would use incremental compilation which caused the metadata to be
// dropped. We now use globals instead and add this test to verify that incremental
// keeps working. Also testing debug mode while at it.

use std::autodiff::autodiff_reverse;

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
