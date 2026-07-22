//@ compile-flags: -Zautodiff=Enable,NoTT -Zautodiff_post_passes=function(mem2reg,instsimplify,simplifycfg) -C opt-level=3  -Clto=fat
//@ no-prefer-dynamic
//@ needs-enzyme

// This test combines two features of Enzyme, automatic differentiation and batching. As such, it is
// especially prone to breakages. I reduced it therefore to a minimal check matches argument/return
// types. Based on the original batching author, implementing the batching feature over MLIR instead
// of LLVM should give significantly more reliable performance.

#![feature(autodiff)]

use std::autodiff::autodiff_forward;

#[autodiff_forward(d_square3, Dual, DualOnly)]
#[autodiff_forward(d_square2, 4, Dual, DualOnly)]
#[autodiff_forward(d_square1, 4, Dual, Dual)]
#[no_mangle]
#[inline(never)]
fn square(x: &f32) -> f32 {
    x * x
}

// CHECK: ; batched::d_square2
// CHECK: define internal fastcc void
// CHECK-SAME: (ptr {{.*}}, ptr {{.*}}, ptr {{.*}}, ptr {{.*}}, ptr {{.*}})
// CHECK:   ret void
// CHECK-NEXT:   }

// CHECK: ; batched::d_square1
// CHECK: define internal fastcc void
// CHECK-SAME: (ptr {{.*}}, ptr {{.*}}, ptr {{.*}}, ptr {{.*}}, ptr {{.*}}, ptr {{.*}})
// CHECK:   ret void
// CHECK-NEXT:   }

// The base ("scalar") case d_square3, without batching.
// CHECK: define internal float @fwddiffesquare(ptr {{.*}}, ptr {{.*}})
// CHECK:  [[SHADOW_X:%"_[0-9]+'ipl"]] = load float, ptr %"x'"
// CHECK-NEXT:  [[PRIMAL_X:%_[0-9]+]] = load float, ptr %x
// CHECK-NEXT:  [[MUL1:%[0-9]+]] = fmul fast float [[SHADOW_X]], [[PRIMAL_X]]
// CHECK-NEXT:  [[MUL2:%[0-9]+]] = fmul fast float [[SHADOW_X]], [[PRIMAL_X]]
// CHECK-NEXT:  [[ADD1:%[0-9]+]] = fadd fast float [[MUL1]], [[MUL2]]
// CHECK-NEXT:  ret float [[ADD1]]
// CHECK-NEXT: }

fn main() {
    let x = std::hint::black_box(3.0);
    let output = square(&x);
    dbg!(&output);
    assert_eq!(9.0, output);
    dbg!(square(&x));

    let mut df_dx1 = 1.0;
    let mut df_dx2 = 2.0;
    let mut df_dx3 = 3.0;
    let mut df_dx4 = 0.0;
    let [o1, o2, o3, o4] = d_square2(&x, &mut df_dx1, &mut df_dx2, &mut df_dx3, &mut df_dx4);
    dbg!(o1, o2, o3, o4);
    let [output2, o1, o2, o3, o4] =
        d_square1(&x, &mut df_dx1, &mut df_dx2, &mut df_dx3, &mut df_dx4);
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
