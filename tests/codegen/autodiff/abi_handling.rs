//@ compile-flags: -Zautodiff=Enable -C opt-level=3 -Clto=fat
//@ no-prefer-dynamic
//@ needs-enzyme

// This does only test the funtion attribute handling for autodiff.
// Function argument changes are troublesome for Enzyme, so we have to
// ensure that arguments remain the same, or if we change them, be aware
// of the changes to handle it correctly.

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

// CHECK: ; abi_handling::f1
// CHECK-NEXT: ; Function Attrs: {{.*}}noinline{{.*}}
// CHECK-NEXT: define dso_local noundef float @_ZN12abi_handling2f1{{.*}}(ptr noalias nocapture noundef readonly align 4 dereferenceable(8) %x)
#[autodiff_forward(df1, Dual, Dual)]
fn f1(x: &[f32; 2]) -> f32 {
    x[0] + x[1]
}

// CHECK: ; abi_handling::f2
// CHECK-NEXT: ; Function Attrs: {{.*}}noinline{{.*}}
// CHECK-NEXT: define dso_local noundef float @_ZN12abi_handling2f217h33732e9f83c91bc9E(ptr nocapture noundef nonnull readonly %f, float noundef %x)
#[autodiff_reverse(df2, Const, Active, Active)]
fn f2(f: fn(f32) -> f32, x: f32) -> f32 {
    f(x)
}

// CHECK: ; abi_handling::f3
// CHECK-NEXT: ; Function Attrs: {{.*}}noinline{{.*}}
// CHECK-NEXT: define dso_local noundef float @_ZN12abi_handling2f317h9cd1fc602b0815a4E(ptr noalias nocapture noundef readonly align 4 dereferenceable(4) %x, ptr noalias nocapture noundef readonly align 4 dereferenceable(4) %y)
#[autodiff_forward(df3, Dual, Dual, Dual)]
fn f3<'a>(x: &'a f32, y: &'a f32) -> f32 {
    *x * *y
}

// CHECK: ; abi_handling::f4
// CHECK-NEXT: ; Function Attrs: {{.*}}noinline{{.*}}
// CHECK-NEXT: define internal fastcc noundef float @_ZN12abi_handling2f417h2f4a9a7492d91e9fE(float noundef %x.0, float noundef %x.1)
#[autodiff_forward(df4, Dual, Dual)]
fn f4(x: (f32, f32)) -> f32 {
    x.0 * x.1
}

// CHECK: ; abi_handling::f5
// CHECK-NEXT: ; Function Attrs: {{.*}}noinline{{.*}}
// CHECK-NEXT: define internal fastcc noundef float @_ZN12abi_handling2f517hf8d4ac4d2c2a3976E(float noundef %i.0, float noundef %i.1)
#[autodiff_forward(df5, Dual, Dual)]
fn f5(i: Input) -> f32 {
    i.x + i.y
}

// CHECK: ; abi_handling::f6
// CHECK-NEXT: ; Function Attrs: {{.*}}noinline{{.*}}
// CHECK-NEXT: define internal fastcc noundef float @_ZN12abi_handling2f617h5784b207bbb2483eE(float noundef %i.0, float noundef %i.1)
#[autodiff_forward(df6, Dual, Dual)]
fn f6(i: NestedInput) -> f32 {
    i.x + i.y.z * i.y.z
}

fn main() {
    let x = std::hint::black_box(2.0);
    let y = std::hint::black_box(3.0);

    let in_f1 = [x, y];
    dbg!(f1(&in_f1));
    let dx1 = std::hint::black_box(&[1.0, 0.0]);
    let res_f1 = df1(&in_f1, dx1);
    dbg!(res_f1);

    dbg!(f2(square, x));
    let res_f2 = df2(square, x, 1.0);
    dbg!(res_f2);

    static Y: f32 = std::hint::black_box(3.2);
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
    let res_f6 = df6(in_f6, NestedInput { x: 1.0, y: Wrapper { z: 0.0 } });
    dbg!(res_f6);
}
