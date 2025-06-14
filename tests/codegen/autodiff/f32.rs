//@ compile-flags: -Zautodiff=Enable -Zautodiff=NoPostopt -C opt-level=3 -Clto=fat -g
//@ no-prefer-dynamic
//@ needs-enzyme
#![feature(autodiff)]

use std::autodiff::autodiff_reverse;

#[autodiff_reverse(d_square, Duplicated, Active)]
#[no_mangle]
fn callee(x: &f32) -> f32 {
    *x * *x
}

fn main() {
    let x: f32 = 7.0;
    let mut df_dx: f32 = 0.0;
    d_square(&x, &mut df_dx, 1.0);
}

// CHECK: define float @callee(ptr align 4 {{.*}}) {{.*}} !dbg {
// CHECK-NEXT: start:
// CHECK: #dbg_value
// CHECK: load float
// CHECK: fmul float
// CHECK: ret float

// CHECK: define void @_ZN9f32_debug8d_square{{.*}}(ptr align 4 {{.*}}, ptr align 4 {{.*}}, float {{.*}}) {{.*}} {
// CHECK: call {{.*}} @diffecallee
// CHECK: ret void

// CHECK: define {{.*}} @diffecallee(ptr {{.*}} align 4 {{.*}}, ptr {{.*}} align 4 {{.*}}, float {{.*}}) {{.*}} {
// CHECK: load float
// CHECK: fmul float
// CHECK: store float
// CHECK: ret {{.*}}
