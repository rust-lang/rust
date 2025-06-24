//@ revisions: debug release

//@[debug] compile-flags: -Zautodiff=Enable -C opt-level=0 -Clto=fat
//@[release] compile-flags: -Zautodiff=Enable -C opt-level=3 -Clto=fat
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
// debug-NEXT: define internal float @_ZN12abi_handling2f117h536ac8081c1e4101E
// debug-SAME: (ptr align 4 %x)
// release-NEXT: define internal fastcc noundef float @_ZN12abi_handling2f117h536ac8081c1e4101E
// release-SAME: (float %x.0.val, float %x.4.val)
#[autodiff_forward(df1, Dual, Dual)]
fn f1(x: &[f32; 2]) -> f32 {
    x[0] + x[1]
}

// CHECK: ; abi_handling::f2
// CHECK-NEXT: ; Function Attrs: {{.*}}noinline{{.*}}
// debug-NEXT: define internal float @_ZN12abi_handling2f217h33732e9f83c91bc9E
// debug-SAME: (ptr %f, float %x)
// release-NEXT: define internal fastcc noundef float @_ZN12abi_handling2f217h33732e9f83c91bc9E
// release-SAME: (float noundef %x)
#[autodiff_reverse(df2, Const, Active, Active)]
fn f2(f: fn(f32) -> f32, x: f32) -> f32 {
    f(x)
}

// CHECK: ; abi_handling::f3
// CHECK-NEXT: ; Function Attrs: {{.*}}noinline{{.*}}
// debug-NEXT: define internal float @_ZN12abi_handling2f317h9cd1fc602b0815a4E
// debug-SAME: (ptr align 4 %x, ptr align 4 %y)
// release-NEXT: define internal fastcc noundef float @_ZN12abi_handling2f317h9cd1fc602b0815a4E
// release-SAME: (float %x.0.val)
#[autodiff_forward(df3, Dual, Dual, Dual)]
fn f3<'a>(x: &'a f32, y: &'a f32) -> f32 {
    *x * *y
}

// CHECK: ; abi_handling::f4
// CHECK-NEXT: ; Function Attrs: {{.*}}noinline{{.*}}
// debug-NEXT: define internal float @_ZN12abi_handling2f417h2f4a9a7492d91e9fE
// debug-SAME: (float %x.0, float %x.1)
// release-NEXT: define internal fastcc noundef float @_ZN12abi_handling2f417h2f4a9a7492d91e9fE
// release-SAME: (float noundef %x.0, float noundef %x.1)
#[autodiff_forward(df4, Dual, Dual)]
fn f4(x: (f32, f32)) -> f32 {
    x.0 * x.1
}

// CHECK: ; abi_handling::f5
// CHECK-NEXT: ; Function Attrs: {{.*}}noinline{{.*}}
// debug-NEXT: define internal float @_ZN12abi_handling2f517hf8d4ac4d2c2a3976E
// debug-SAME: (float %i.0, float %i.1)
// release-NEXT: define internal fastcc noundef float @_ZN12abi_handling2f517hf8d4ac4d2c2a3976E
// release-SAME: (float noundef %i.0, float noundef %i.1)
#[autodiff_forward(df5, Dual, Dual)]
fn f5(i: Input) -> f32 {
    i.x + i.y
}

// CHECK: ; abi_handling::f6
// CHECK-NEXT: ; Function Attrs: {{.*}}noinline{{.*}}
// debug-NEXT: define internal float @_ZN12abi_handling2f617h5784b207bbb2483eE
// debug-SAME: (float %i.0, float %i.1)
// release-NEXT: define internal fastcc noundef float @_ZN12abi_handling2f617h5784b207bbb2483eE
// release-SAME: (float noundef %i.0, float noundef %i.1)
#[autodiff_forward(df6, Dual, Dual)]
fn f6(i: NestedInput) -> f32 {
    i.x + i.y.z * i.y.z
}

// df1
// release: define internal fastcc { float, float }
// release-SAME: @fwddiffe_ZN12abi_handling2f117h536ac8081c1e4101E
// release-SAME: (float %x.0.val, float %x.4.val)
// release-NEXT: start:
// release-NEXT: %_0 = fadd float %x.0.val, %x.4.val
// release-NEXT: %0 = insertvalue { float, float } undef, float %_0, 0
// release-NEXT: %1 = insertvalue { float, float } %0, float 1.000000e+00, 1
// release-NEXT: ret { float, float } %1
// release-NEXT: }

// debug: define internal { float, float } @fwddiffe_ZN12abi_handling2f117h536ac8081c1e4101E
// debug-SAME: (ptr align 4 %x, ptr align 4 %"x'")
// debug-NEXT: start:
// debug-NEXT: %"'ipg" = getelementptr inbounds float, ptr %"x'", i64 0
// debug-NEXT: %0 = getelementptr inbounds nuw float, ptr %x, i64 0
// debug-NEXT: %"_2'ipl" = load float, ptr %"'ipg", align 4, !alias.scope !4, !noalias !7
// debug-NEXT: %_2 = load float, ptr %0, align 4, !alias.scope !7, !noalias !4
// debug-NEXT: %"'ipg2" = getelementptr inbounds float, ptr %"x'", i64 1
// debug-NEXT: %1 = getelementptr inbounds nuw float, ptr %x, i64 1
// debug-NEXT: %"_5'ipl" = load float, ptr %"'ipg2", align 4, !alias.scope !4, !noalias !7
// debug-NEXT: %_5 = load float, ptr %1, align 4, !alias.scope !7, !noalias !4
// debug-NEXT: %_0 = fadd float %_2, %_5
// debug-NEXT: %2 = fadd fast float %"_2'ipl", %"_5'ipl"
// debug-NEXT: %3 = insertvalue { float, float } undef, float %_0, 0
// debug-NEXT: %4 = insertvalue { float, float } %3, float %2, 1
// debug-NEXT: ret { float, float } %4
// debug-NEXT: }

// df2
// release: define internal fastcc { float, float }
// release-SAME: @diffe_ZN12abi_handling2f217h33732e9f83c91bc9E
// release-SAME: (float noundef %x)
// release-NEXT: invertstart:
// release-NEXT: %_0.i = fmul float %x, %x
// release-NEXT: %0 = insertvalue { float, float } undef, float %_0.i, 0
// release-NEXT: %1 = insertvalue { float, float } %0, float 0.000000e+00, 1
// release-NEXT: ret { float, float } %1
// release-NEXT: }

// debug: define internal { float, float } @diffe_ZN12abi_handling2f217h33732e9f83c91bc9E
// debug-SAME: (ptr %f, float %x, float %differeturn)
// debug-NEXT: start:
// debug-NEXT: %"x'de" = alloca float, align 4
// debug-NEXT: store float 0.000000e+00, ptr %"x'de", align 4
// debug-NEXT: %toreturn = alloca float, align 4
// debug-NEXT: %_0 = call float %f(float %x) #12
// debug-NEXT: store float %_0, ptr %toreturn, align 4
// debug-NEXT: br label %invertstart
// debug-EMPTY:
// debug-NEXT: invertstart:                                      ; preds = %start
// debug-NEXT: %retreload = load float, ptr %toreturn, align 4
// debug-NEXT: %0 = load float, ptr %"x'de", align 4
// debug-NEXT: %1 = insertvalue { float, float } undef, float %retreload, 0
// debug-NEXT: %2 = insertvalue { float, float } %1, float %0, 1
// debug-NEXT: ret { float, float } %2
// debug-NEXT: }

// df3
// release: define internal fastcc { float, float }
// release-SAME: @fwddiffe_ZN12abi_handling2f317h9cd1fc602b0815a4E
// release-SAME: (float %x.0.val)
// release-NEXT: start:
// release-NEXT: %0 = insertvalue { float, float } undef, float %x.0.val, 0
// release-NEXT: %1 = insertvalue { float, float } %0, float 0x40099999A0000000, 1
// release-NEXT: ret { float, float } %1
// release-NEXT: }

// debug: define internal { float, float } @fwddiffe_ZN12abi_handling2f317h9cd1fc602b0815a4E
// debug-SAME: (ptr align 4 %x, ptr align 4 %"x'", ptr align 4 %y, ptr align 4 %"y'")
// debug-NEXT: start:
// debug-NEXT: %"_3'ipl" = load float, ptr %"x'", align 4, !alias.scope !9, !noalias !12
// debug-NEXT: %_3 = load float, ptr %x, align 4, !alias.scope !12, !noalias !9
// debug-NEXT: %"_4'ipl" = load float, ptr %"y'", align 4, !alias.scope !14, !noalias !17
// debug-NEXT: %_4 = load float, ptr %y, align 4, !alias.scope !17, !noalias !14
// debug-NEXT: %_0 = fmul float %_3, %_4
// debug-NEXT: %0 = fmul fast float %"_3'ipl", %_4
// debug-NEXT: %1 = fmul fast float %"_4'ipl", %_3
// debug-NEXT: %2 = fadd fast float %0, %1
// debug-NEXT: %3 = insertvalue { float, float } undef, float %_0, 0
// debug-NEXT: %4 = insertvalue { float, float } %3, float %2, 1
// debug-NEXT: ret { float, float } %4
// debug-NEXT: }

// df4
// release: define internal fastcc { float, float }
// release-SAME: @fwddiffe_ZN12abi_handling2f417h2f4a9a7492d91e9fE
// release-SAME: (float noundef %x.0, float %"x.0'")
// release-NEXT: start:
// release-NEXT: %0 = insertvalue { float, float } undef, float %x.0, 0
// release-NEXT: %1 = insertvalue { float, float } %0, float %"x.0'", 1
// release-NEXT: ret { float, float } %1
// release-NEXT: }

// debug: define internal { float, float } @fwddiffe_ZN12abi_handling2f417h2f4a9a7492d91e9fE
// debug-SAME: (float %x.0, float %"x.0'", float %x.1, float %"x.1'")
// debug-NEXT: start:
// debug-NEXT: %_0 = fmul float %x.0, %x.1
// debug-NEXT: %0 = fmul fast float %"x.0'", %x.1
// debug-NEXT: %1 = fmul fast float %"x.1'", %x.0
// debug-NEXT: %2 = fadd fast float %0, %1
// debug-NEXT: %3 = insertvalue { float, float } undef, float %_0, 0
// debug-NEXT: %4 = insertvalue { float, float } %3, float %2, 1
// debug-NEXT: ret { float, float } %4
// debug-NEXT: }

// df5
// release: define internal fastcc { float, float }
// release-SAME: @fwddiffe_ZN12abi_handling2f517hf8d4ac4d2c2a3976E
// release-SAME: (float noundef %i.0, float %"i.0'")
// release-NEXT: start:
// release-NEXT: %_0 = fadd float %i.0, 1.000000e+00
// release-NEXT: %0 = insertvalue { float, float } undef, float %_0, 0
// release-NEXT: %1 = insertvalue { float, float } %0, float %"i.0'", 1
// release-NEXT: ret { float, float } %1
// release-NEXT: }

// debug: define internal { float, float } @fwddiffe_ZN12abi_handling2f517hf8d4ac4d2c2a3976E
// debug-SAME: (float %i.0, float %"i.0'", float %i.1, float %"i.1'")
// debug-NEXT: start:
// debug-NEXT: %_0 = fadd float %i.0, %i.1
// debug-NEXT: %0 = fadd fast float %"i.0'", %"i.1'"
// debug-NEXT: %1 = insertvalue { float, float } undef, float %_0, 0
// debug-NEXT: %2 = insertvalue { float, float } %1, float %0, 1
// debug-NEXT: ret { float, float } %2
// debug-NEXT: }

// df6
// release: define internal fastcc { float, float }
// release-SAME: @fwddiffe_ZN12abi_handling2f617h5784b207bbb2483eE
// release-SAME: (float noundef %i.0, float %"i.0'", float noundef %i.1, float %"i.1'")
// release-NEXT: start:
// release-NEXT: %_3 = fmul float %i.1, %i.1
// release-NEXT: %0 = fadd fast float %"i.1'", %"i.1'"
// release-NEXT: %1 = fmul fast float %0, %i.1
// release-NEXT: %_0 = fadd float %i.0, %_3
// release-NEXT: %2 = fadd fast float %"i.0'", %1
// release-NEXT: %3 = insertvalue { float, float } undef, float %_0, 0
// release-NEXT: %4 = insertvalue { float, float } %3, float %2, 1
// release-NEXT: ret { float, float } %4
// release-NEXT: }

// debug: define internal { float, float } @fwddiffe_ZN12abi_handling2f617h5784b207bbb2483eE
// debug-SAME: (float %i.0, float %"i.0'", float %i.1, float %"i.1'")
// debug-NEXT: start:
// debug-NEXT: %_3 = fmul float %i.1, %i.1
// debug-NEXT: %0 = fmul fast float %"i.1'", %i.1
// debug-NEXT: %1 = fmul fast float %"i.1'", %i.1
// debug-NEXT: %2 = fadd fast float %0, %1
// debug-NEXT: %_0 = fadd float %i.0, %_3
// debug-NEXT: %3 = fadd fast float %"i.0'", %2
// debug-NEXT: %4 = insertvalue { float, float } undef, float %_0, 0
// debug-NEXT: %5 = insertvalue { float, float } %4, float %3, 1
// debug-NEXT: ret { float, float } %5
// debug-NEXT: }

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
}
