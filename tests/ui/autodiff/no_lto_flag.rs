//@ needs-enzyme
//@ no-prefer-dynamic
//@ revisions: with_lto no_lto
//@[with_lto] compile-flags: -Zautodiff=Enable -C opt-level=3  -Clto=fat
//@[no_lto] compile-flags: -Zautodiff=Enable -C opt-level=3 -Clto=thin

#![feature(autodiff)]
//@[no_lto] build-fail
//@[with_lto] build-pass

// Autodiff requires users to enable lto=fat (for now).
// In the past, autodiff did not run if users forget to enable fat-lto, which caused functions to
// returning zero-derivatives. That's obviously wrong and confusing to users. We now added a check
// which will abort compilation instead.

use std::autodiff::autodiff_reverse;
//[no_lto]~? ERROR using the autodiff feature requires setting `lto="fat"` in your Cargo.toml

#[autodiff_reverse(d_square, Duplicated, Active)]
fn square(x: &f64) -> f64 {
    *x * *x
}

fn main() {
    let xf64: f64 = std::hint::black_box(3.0);

    let mut df_dxf64: f64 = std::hint::black_box(0.0);

    let _output_f64 = d_square(&xf64, &mut df_dxf64, 1.0);
    assert_eq!(6.0, df_dxf64);
}
