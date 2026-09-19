//@ compile-flags: -Zautodiff=Enable,NoTT -Zautodiff_post_passes=function(mem2reg,instsimplify,simplifycfg) -C opt-level=3  -Clto=fat
//@ no-prefer-dynamic
//@ needs-enzyme
#![feature(autodiff)]

use std::autodiff::autodiff_reverse;

#[autodiff_reverse(d_square, Duplicated, Active)]
#[no_mangle]
#[inline(never)]
fn square(x: &f64) -> f64 {
    x * x
}

// CHECK:define internal { double } @diffesquare(ptr {{.*}}, ptr {{.*}}, double {{.*}})
// CHECK-NEXT:start:
// CHECK-NEXT:  [[X:%_[0-9]+]] = load double, ptr %x, align 8
// CHECK-NEXT:  [[SQUARE:%_[0-9]+]] = fmul double [[X]], [[X]]
// CHECK-NEXT:  [[DIFFR1:%[0-9]+]] = fmul fast double %differeturn, [[X]]
// CHECK-NEXT:  [[DIFFR2:%[0-9]+]] = fmul fast double %differeturn, [[X]]
// CHECK-NEXT:  [[ADD1:%[0-9]+]] = fadd fast double [[DIFFR1]], [[DIFFR2]]
// CHECK-NEXT:  [[SHADOW_X:%[0-9]+]] = load double, ptr %"x'", align 8
// CHECK-NEXT:  [[ADD2:%[0-9]+]] = fadd fast double [[SHADOW_X]], [[ADD1]]
// CHECK-NEXT:  store double [[ADD2]], ptr %"x'", align 8
// CHECK-NEXT:  [[RET:%[0-9]+]] = insertvalue { double } undef, double [[SQUARE]], 0
// CHECK-NEXT:  ret { double } [[RET]]
// CHECK-NEXT:}

fn main() {
    let x = std::hint::black_box(3.0);
    let output = square(&x);
    assert_eq!(9.0, output);

    let mut df_dx = 0.0;
    let output_ = d_square(&x, &mut df_dx, 1.0);
    assert_eq!(output, output_);
    assert_eq!(6.0, df_dx);
}
