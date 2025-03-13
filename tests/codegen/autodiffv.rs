//@ compile-flags: -Zautodiff=Enable -C opt-level=3  -Clto=fat
//@ no-prefer-dynamic
//@ needs-enzyme

#![feature(autodiff)]

use std::autodiff::autodiff;

#[autodiff(d_square3, Forward, Dual, DualOnly)]
#[no_mangle]
fn squaref(x: &f32) -> f32 {
    2.0 * x * x
}


#[autodiff(d_square2, Forward, 4, Dual, DualOnly)]
#[autodiff(d_square, Forward, 4, Dual, Dual)]
#[no_mangle]
fn square(x: &f32) -> f32 {
    x * x
}

// CHECK:define internal fastcc void @diffe4square([4 x ptr] %"x'"
// CHECK-NEXT:invertstart:
// CHECK-NEXT:  %0 = extractvalue [4 x ptr] %"x'", 0
// CHECK-NEXT:  %1 = load double, ptr %0, align 8, !alias.scope !15950, !noalias !15953
// CHECK-NEXT:  %2 = fadd fast double %1, 6.000000e+00
// CHECK-NEXT:  store double %2, ptr %0, align 8, !alias.scope !15950, !noalias !15953
// CHECK-NEXT:  %3 = extractvalue [4 x ptr] %"x'", 1
// CHECK-NEXT:  %4 = load double, ptr %3, align 8, !alias.scope !15958, !noalias !15959
// CHECK-NEXT:  %5 = fadd fast double %4, 6.000000e+00
// CHECK-NEXT:  store double %5, ptr %3, align 8, !alias.scope !15958, !noalias !15959
// CHECK-NEXT:  %6 = extractvalue [4 x ptr] %"x'", 2
// CHECK-NEXT:  %7 = load double, ptr %6, align 8, !alias.scope !15960, !noalias !15961
// CHECK-NEXT:  %8 = fadd fast double %7, 6.000000e+00
// CHECK-NEXT:  store double %8, ptr %6, align 8, !alias.scope !15960, !noalias !15961
// CHECK-NEXT:  %9 = extractvalue [4 x ptr] %"x'", 3
// CHECK-NEXT:  %10 = load double, ptr %9, align 8, !alias.scope !15962, !noalias !15963
// CHECK-NEXT:  %11 = fadd fast double %10, 6.000000e+00
// CHECK-NEXT:  store double %11, ptr %9, align 8, !alias.scope !15962, !noalias !15963
// CHECK-NEXT:  ret void
// CHECK-NEXT:}

fn main() {
    let x = std::hint::black_box(3.0);
    let output = square(&x);
    dbg!(&output);
    assert_eq!(9.0, output);
    dbg!(squaref(&x));

    let mut df_dx1 = 1.0;
    let mut df_dx2 = 2.0;
    let mut df_dx3 = 3.0;
    let mut df_dx4 = 0.0;
    let [o1,o2,o3,o4] = d_square2(&x, &mut df_dx1, &mut df_dx2, &mut df_dx3,  &mut df_dx4);
    dbg!(o1, o2, o3, o4);
    let [output2, o1,o2,o3,o4] = d_square(&x, &mut df_dx1, &mut df_dx2, &mut df_dx3,  &mut df_dx4);
    dbg!(o1, o2, o3, o4);
    assert_eq!(output, output2);
    assert!((6.0 - o1).abs() < 1e-10);
    assert!((12.0 - o2).abs() < 1e-10);
    assert!((18.0 - o3).abs() < 1e-10);
    assert!((0.0 - o4).abs() < 1e-10);
    assert_eq!(1.0, df_dx1);
    assert_eq!(2.0, df_dx2);
    assert_eq!(3.0, df_dx3);
    assert_eq!(0.0, df_dx4);
    assert_eq!(d_square3(&x, &mut df_dx1), 2.0 * o1);
    assert_eq!(d_square3(&x, &mut df_dx2), 2.0 * o2);
    assert_eq!(d_square3(&x, &mut df_dx3), 2.0 * o3);
    assert_eq!(d_square3(&x, &mut df_dx4), 2.0 * o4);
}
