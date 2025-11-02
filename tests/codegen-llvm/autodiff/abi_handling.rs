//@ revisions: debug release

//@[debug] compile-flags: -Zautodiff=Enable,NoTT -C opt-level=0 -Clto=fat
//@[release] compile-flags: -Zautodiff=Enable,NoTT -C opt-level=3 -Clto=fat
//@ no-prefer-dynamic
//@ needs-enzyme

// This test checks that Rust types are lowered to LLVM-IR types in a way
// we expect and Enzyme can handle. We explicitly check release mode to
// ensure that LLVM's O3 pipeline doesn't rewrite function signatures
// into forms that Enzyme can't process correctly.

#![feature(autodiff)]

use std::autodiff::{autodiff_forward, autodiff_reverse};

#[derive(Copy, Clone)]
struct Input {
    x: f32,
    y: f32,
}

#[derive(Copy, Clone)]
struct Wrapper {
    z: f32,
}

#[derive(Copy, Clone)]
struct NestedInput {
    x: f32,
    y: Wrapper,
}

fn square(x: f32) -> f32 {
    x * x
}

// CHECK-LABEL: ; abi_handling::df1
// CHECK-NEXT: Function Attrs
// debug-NEXT: define internal { float, float }
// debug-SAME: (ptr align 4 %x, ptr align 4 %bx_0)
// release-NEXT: define internal fastcc float
// release-SAME: (float %x.0.val, float %x.4.val)

// CHECK-LABEL: ; abi_handling::f1
// CHECK-NEXT: Function Attrs
// debug-NEXT: define internal float
// debug-SAME: (ptr align 4 %x)
// release-NEXT: define internal fastcc noundef float
// release-SAME: (float %x.0.val, float %x.4.val)
#[autodiff_forward(df1, Dual, Dual)]
#[inline(never)]
fn f1(x: &[f32; 2]) -> f32 {
    x[0] + x[1]
}

// CHECK-LABEL: ; abi_handling::df2
// CHECK-NEXT: Function Attrs
// debug-NEXT: define internal { float, float }
// debug-SAME: (ptr %f, float %x, float %dret)
// release-NEXT: define internal fastcc float
// release-SAME: (float noundef %x)

// CHECK-LABEL: ; abi_handling::f2
// CHECK-NEXT: Function Attrs
// debug-NEXT: define internal float
// debug-SAME: (ptr %f, float %x)
// release-NEXT: define internal fastcc noundef float
// release-SAME: (float noundef %x)
#[autodiff_reverse(df2, Const, Active, Active)]
#[inline(never)]
fn f2(f: fn(f32) -> f32, x: f32) -> f32 {
    f(x)
}

// CHECK-LABEL: ; abi_handling::df3
// CHECK-NEXT: Function Attrs
// debug-NEXT: define internal { float, float }
// debug-SAME: (ptr align 4 %x, ptr align 4 %bx_0, ptr align 4 %y, ptr align 4 %by_0)
// release-NEXT: define internal fastcc { float, float }
// release-SAME: (float %x.0.val)

// CHECK-LABEL: ; abi_handling::f3
// CHECK-NEXT: Function Attrs
// debug-NEXT: define internal float
// debug-SAME: (ptr align 4 %x, ptr align 4 %y)
// release-NEXT: define internal fastcc noundef float
// release-SAME: (float %x.0.val)
#[autodiff_forward(df3, Dual, Dual, Dual)]
#[inline(never)]
fn f3<'a>(x: &'a f32, y: &'a f32) -> f32 {
    *x * *y
}

// CHECK-LABEL: ; abi_handling::df4
// CHECK-NEXT: Function Attrs
// debug-NEXT: define internal { float, float }
// debug-SAME: (float %x.0, float %x.1, float %bx_0.0, float %bx_0.1)
// release-NEXT: define internal fastcc { float, float }
// release-SAME: (float noundef %x.0, float noundef %x.1)

// CHECK-LABEL: ; abi_handling::f4
// CHECK-NEXT: Function Attrs
// debug-NEXT: define internal float
// debug-SAME: (float %x.0, float %x.1)
// release-NEXT: define internal fastcc noundef float
// release-SAME: (float noundef %x.0, float noundef %x.1)
#[autodiff_forward(df4, Dual, Dual)]
#[inline(never)]
fn f4(x: (f32, f32)) -> f32 {
    x.0 * x.1
}

// CHECK-LABEL: ; abi_handling::df5
// CHECK-NEXT: Function Attrs
// debug-NEXT: define internal { float, float }
// debug-SAME: (float %i.0, float %i.1, float %bi_0.0, float %bi_0.1)
// release-NEXT: define internal fastcc { float, float }
// release-SAME: (float noundef %i.0, float noundef %i.1)

// CHECK-LABEL: ; abi_handling::f5
// CHECK-NEXT: Function Attrs
// debug-NEXT: define internal float
// debug-SAME: (float %i.0, float %i.1)
// release-NEXT: define internal fastcc noundef float
// release-SAME: (float noundef %i.0, float noundef %i.1)
#[autodiff_forward(df5, Dual, Dual)]
#[inline(never)]
fn f5(i: Input) -> f32 {
    i.x + i.y
}

// CHECK-LABEL: ; abi_handling::df6
// CHECK-NEXT: Function Attrs
// debug-NEXT: define internal { float, float }
// debug-SAME: (float %i.0, float %i.1, float %bi_0.0, float %bi_0.1)
// release-NEXT: define internal fastcc { float, float }
// release-SAME: float noundef %i.0, float noundef %i.1
// release-SAME: float noundef %bi_0.0, float noundef %bi_0.1

// CHECK-LABEL: ; abi_handling::f6
// CHECK-NEXT: Function Attrs
// debug-NEXT: define internal float
// debug-SAME: (float %i.0, float %i.1)
// release-NEXT: define internal fastcc noundef float
// release-SAME: (float noundef %i.0, float noundef %i.1)
#[autodiff_forward(df6, Dual, Dual)]
#[inline(never)]
fn f6(i: NestedInput) -> f32 {
    i.x + i.y.z * i.y.z
}

// CHECK-LABEL: ; abi_handling::df7
// CHECK-NEXT: Function Attrs
// debug-NEXT: define internal { float, float }
// debug-SAME: (ptr align 4 %x.0, ptr align 4 %x.1, ptr align 4 %bx_0.0, ptr align 4 %bx_0.1)
// release-NEXT: define internal fastcc { float, float }
// release-SAME: (float %x.0.0.val, float %x.1.0.val)

// CHECK-LABEL: ; abi_handling::f7
// CHECK-NEXT: Function Attrs
// debug-NEXT: define internal float
// debug-SAME: (ptr align 4 %x.0, ptr align 4 %x.1)
// release-NEXT: define internal fastcc noundef float
// release-SAME: (float %x.0.0.val, float %x.1.0.val)
#[autodiff_forward(df7, Dual, Dual)]
#[inline(never)]
fn f7(x: (&f32, &f32)) -> f32 {
    x.0 * x.1
}

fn main() {
    let x = std::hint::black_box(2.0);
    let y = std::hint::black_box(3.0);
    let z = std::hint::black_box(4.0);
    static Y: f32 = std::hint::black_box(3.2);

    let in_f1 = [x, y];
    dbg!(f1(&in_f1));
    let res_f1 = df1(&in_f1, &[1.0, 0.0]);
    dbg!(res_f1);

    dbg!(f2(square, x));
    let res_f2 = df2(square, x, 1.0);
    dbg!(res_f2);

    dbg!(f3(&x, &Y));
    let res_f3 = df3(&x, &Y, &1.0, &0.0);
    dbg!(res_f3);

    let in_f4 = (x, y);
    dbg!(f4(in_f4));
    let res_f4 = df4(in_f4, (1.0, 0.0));
    dbg!(res_f4);

    let in_f5 = Input { x, y };
    dbg!(f5(in_f5));
    let res_f5 = df5(in_f5, Input { x: 1.0, y: 0.0 });
    dbg!(res_f5);

    let in_f6 = NestedInput { x, y: Wrapper { z: y } };
    dbg!(f6(in_f6));
    let res_f6 = df6(in_f6, NestedInput { x, y: Wrapper { z } });
    dbg!(res_f6);

    let in_f7 = (&x, &y);
    dbg!(f7(in_f7));
    let res_f7 = df7(in_f7, (&1.0, &0.0));
    dbg!(res_f7);
}
