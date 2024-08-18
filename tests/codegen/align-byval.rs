// ignore-tidy-linelength
//@ revisions:m68k wasm x86_64-linux x86_64-windows i686-linux i686-windows

//@[m68k] compile-flags: --target m68k-unknown-linux-gnu
//@[m68k] needs-llvm-components: m68k
//@[wasm] compile-flags: --target wasm32-unknown-emscripten
//@[wasm] needs-llvm-components: webassembly
//@[x86_64-linux] compile-flags: --target x86_64-unknown-linux-gnu
//@[x86_64-linux] needs-llvm-components: x86
//@[x86_64-windows] compile-flags: --target x86_64-pc-windows-msvc
//@[x86_64-windows] needs-llvm-components: x86
//@[i686-linux] compile-flags: --target i686-unknown-linux-gnu
//@[i686-linux] needs-llvm-components: x86
//@[i686-windows] compile-flags: --target i686-pc-windows-msvc
//@[i686-windows] needs-llvm-components: x86

// Tests that `byval` alignment is properly specified (#80127).
// The only targets that use `byval` are m68k, wasm, x86-64, and x86.
// Note also that Windows mandates a by-ref ABI here, so it does not use byval.

#![feature(no_core, lang_items)]
#![crate_type = "lib"]
#![no_std]
#![no_core]

#[lang = "sized"]
trait Sized {}
#[lang = "freeze"]
trait Freeze {}
#[lang = "copy"]
trait Copy {}

impl Copy for i32 {}
impl Copy for i64 {}

// This struct can be represented as a pair, so it exercises the OperandValue::Pair
// codepath in `codegen_argument`.
#[repr(C)]
pub struct NaturalAlign1 {
    a: i8,
    b: i8,
}

// This struct cannot be represented as an immediate, so it exercises the OperandValue::Ref
// codepath in `codegen_argument`.
#[repr(C)]
pub struct NaturalAlign2 {
    a: [i16; 16],
    b: i16,
}

#[repr(C)]
#[repr(align(4))]
pub struct ForceAlign4 {
    a: [i8; 16],
    b: i8,
}

// On i686-windows, this is passed on stack using `byval`
#[repr(C)]
pub struct NaturalAlign8 {
    a: i64,
    b: i64,
    c: i64,
}

// On i686-windows, this is passed by reference (because alignment is >4 and requested/forced),
// even though it has the exact same layout as `NaturalAlign8`!
#[repr(C)]
#[repr(align(8))]
pub struct ForceAlign8 {
    a: i64,
    b: i64,
    c: i64,
}

// On i686-windows, this is passed on stack, because requested alignment is <=4.
#[repr(C)]
#[repr(align(4))]
pub struct LowerFA8 {
    a: i64,
    b: i64,
    c: i64,
}

// On i686-windows, this is passed by reference, because it contains a field with
// requested/forced alignment.
#[repr(C)]
pub struct WrappedFA8 {
    a: ForceAlign8,
}

// On i686-windows, this has the same ABI as ForceAlign8, i.e. passed by reference.
#[repr(transparent)]
pub struct TransparentFA8 {
    _0: (),
    a: ForceAlign8,
}

#[repr(C)]
#[repr(align(16))]
pub struct ForceAlign16 {
    a: [i32; 16],
    b: i8,
}

// CHECK-LABEL: @call_na1
#[no_mangle]
pub unsafe fn call_na1(x: NaturalAlign1) {
    extern "C" {
        fn natural_align_1(x: NaturalAlign1);
    }

    // CHECK: start:

    // m68k: [[ALLOCA:%[a-z0-9+]]] = alloca [2 x i8], align 1
    // m68k: call void @natural_align_1({{.*}}byval([2 x i8]) align 1{{.*}} [[ALLOCA]])

    // wasm: [[ALLOCA:%[a-z0-9+]]] = alloca [2 x i8], align 1
    // wasm: call void @natural_align_1({{.*}}byval([2 x i8]) align 1{{.*}} [[ALLOCA]])

    // x86_64-linux: call void @natural_align_1(i16 %

    // x86_64-windows: call void @natural_align_1(i16 %

    // i686-linux: [[ALLOCA:%[a-z0-9+]]] = alloca [2 x i8], align 4
    // i686-linux: call void @natural_align_1({{.*}}byval([2 x i8]) align 4{{.*}} [[ALLOCA]])

    // i686-windows: [[ALLOCA:%[a-z0-9+]]] = alloca [2 x i8], align 4
    // i686-windows: call void @natural_align_1({{.*}}byval([2 x i8]) align 4{{.*}} [[ALLOCA]])
    natural_align_1(x);
}

// CHECK-LABEL: @call_na2
#[no_mangle]
pub unsafe fn call_na2(x: NaturalAlign2) {
    extern "C" {
        fn natural_align_2(x: NaturalAlign2);
    }

    // CHECK: start:

    // m68k-NEXT: call void @natural_align_2({{.*}}byval([34 x i8]) align 2{{.*}})
    // wasm-NEXT: call void @natural_align_2({{.*}}byval([34 x i8]) align 2{{.*}})
    // x86_64-linux-NEXT: call void @natural_align_2({{.*}}byval([34 x i8]) align 2{{.*}})

    // x86_64-windows-NEXT: call void @natural_align_2
    // x86_64-windows-NOT: byval
    // x86_64-windows-SAME: align 2{{.*}})

    // i686-linux: [[ALLOCA:%[0-9]+]] = alloca [34 x i8], align 4
    // i686-linux: call void @natural_align_2({{.*}}byval([34 x i8]) align 4{{.*}} [[ALLOCA]])

    // i686-windows: [[ALLOCA:%[0-9]+]] = alloca [34 x i8], align 4
    // i686-windows: call void @natural_align_2({{.*}}byval([34 x i8]) align 4{{.*}} [[ALLOCA]])
    natural_align_2(x);
}

// CHECK-LABEL: @call_fa4
#[no_mangle]
pub unsafe fn call_fa4(x: ForceAlign4) {
    extern "C" {
        fn force_align_4(x: ForceAlign4);
    }

    // CHECK: start:
    // m68k-NEXT: call void @force_align_4({{.*}}byval([20 x i8]) align 4{{.*}})

    // wasm-NEXT: call void @force_align_4({{.*}}byval([20 x i8]) align 4{{.*}})

    // x86_64-linux-NEXT: call void @force_align_4({{.*}}byval([20 x i8]) align 4{{.*}})

    // x86_64-windows-NEXT: call void @force_align_4(
    // x86_64-windows-NOT: byval
    // x86_64-windows-SAME: align 4{{.*}})

    // i686-linux-NEXT: call void @force_align_4({{.*}}byval([20 x i8]) align 4{{.*}})

    // i686-windows-NEXT: call void @force_align_4({{.*}}byval([20 x i8]) align 4{{.*}})
    force_align_4(x);
}

// CHECK-LABEL: @call_na8
#[no_mangle]
pub unsafe fn call_na8(x: NaturalAlign8) {
    extern "C" {
        fn natural_align_8(x: NaturalAlign8);
    }

    // CHECK: start:
    // m68k-NEXT: call void @natural_align_8({{.*}}byval([24 x i8]) align 4{{.*}})

    // wasm-NEXT: call void @natural_align_8({{.*}}byval([24 x i8]) align 8{{.*}})

    // x86_64-linux-NEXT: call void @natural_align_8({{.*}}byval([24 x i8]) align 8{{.*}})

    // x86_64-windows-NEXT: call void @natural_align_8(
    // x86_64-windows-NOT: byval
    // x86_64-windows-SAME: align 8{{.*}})

    // i686-linux-NEXT: call void @natural_align_8({{.*}}byval([24 x i8]) align 4{{.*}})

    // i686-windows-NEXT: call void @natural_align_8({{.*}}byval([24 x i8]) align 4{{.*}})
    natural_align_8(x);
}

// CHECK-LABEL: @call_fa8
#[no_mangle]
pub unsafe fn call_fa8(x: ForceAlign8) {
    extern "C" {
        fn force_align_8(x: ForceAlign8);
    }

    // CHECK: start:
    // m68k-NEXT: call void @force_align_8({{.*}}byval([24 x i8]) align 8{{.*}})

    // wasm-NEXT: call void @force_align_8({{.*}}byval([24 x i8]) align 8{{.*}})

    // x86_64-linux-NEXT: call void @force_align_8({{.*}}byval([24 x i8]) align 8{{.*}})

    // x86_64-windows-NEXT: call void @force_align_8(
    // x86_64-windows-NOT: byval
    // x86_64-windows-SAME: align 8{{.*}})

    // i686-linux-NEXT: call void @force_align_8({{.*}}byval([24 x i8]) align 4{{.*}})

    // i686-windows-NEXT: call void @force_align_8(
    // i686-windows-NOT: byval
    // i686-windows-SAME: align 8{{.*}})
    force_align_8(x);
}

// CHECK-LABEL: @call_lfa8
#[no_mangle]
pub unsafe fn call_lfa8(x: LowerFA8) {
    extern "C" {
        fn lower_fa8(x: LowerFA8);
    }

    // CHECK: start:
    // m68k-NEXT: call void @lower_fa8({{.*}}byval([24 x i8]) align 4{{.*}})

    // wasm-NEXT: call void @lower_fa8({{.*}}byval([24 x i8]) align 8{{.*}})

    // x86_64-linux-NEXT: call void @lower_fa8({{.*}}byval([24 x i8]) align 8{{.*}})

    // x86_64-windows-NEXT: call void @lower_fa8(
    // x86_64-windows-NOT: byval
    // x86_64-windows-SAME: align 8{{.*}})

    // i686-linux-NEXT: call void @lower_fa8({{.*}}byval([24 x i8]) align 4{{.*}})

    // i686-windows-NEXT: call void @lower_fa8({{.*}}byval([24 x i8]) align 4{{.*}})
    lower_fa8(x);
}

// CHECK-LABEL: @call_wfa8
#[no_mangle]
pub unsafe fn call_wfa8(x: WrappedFA8) {
    extern "C" {
        fn wrapped_fa8(x: WrappedFA8);
    }

    // CHECK: start:
    // m68k-NEXT: call void @wrapped_fa8({{.*}}byval([24 x i8]) align 8{{.*}})

    // wasm-NEXT: call void @wrapped_fa8({{.*}}byval([24 x i8]) align 8{{.*}})

    // x86_64-linux-NEXT: call void @wrapped_fa8({{.*}}byval([24 x i8]) align 8{{.*}})

    // x86_64-windows-NEXT: call void @wrapped_fa8(
    // x86_64-windows-NOT: byval
    // x86_64-windows-SAME: align 8{{.*}})

    // i686-linux-NEXT: call void @wrapped_fa8({{.*}}byval([24 x i8]) align 4{{.*}})

    // i686-windows-NEXT: call void @wrapped_fa8(
    // i686-windows-NOT: byval
    // i686-windows-SAME: align 8{{.*}})
    wrapped_fa8(x);
}

// CHECK-LABEL: @call_tfa8
#[no_mangle]
pub unsafe fn call_tfa8(x: TransparentFA8) {
    extern "C" {
        fn transparent_fa8(x: TransparentFA8);
    }

    // CHECK: start:
    // m68k-NEXT: call void @transparent_fa8({{.*}}byval([24 x i8]) align 8{{.*}})

    // wasm-NEXT: call void @transparent_fa8({{.*}}byval([24 x i8]) align 8{{.*}})

    // x86_64-linux-NEXT: call void @transparent_fa8({{.*}}byval([24 x i8]) align 8{{.*}})

    // x86_64-windows-NEXT: call void @transparent_fa8(
    // x86_64-windows-NOT: byval
    // x86_64-windows-SAME: align 8{{.*}})

    // i686-linux-NEXT: call void @transparent_fa8({{.*}}byval([24 x i8]) align 4{{.*}})

    // i686-windows-NEXT: call void @transparent_fa8(
    // i686-windows-NOT: byval
    // i686-windows-SAME: align 8{{.*}})
    transparent_fa8(x);
}

// CHECK-LABEL: @call_fa16
#[no_mangle]
pub unsafe fn call_fa16(x: ForceAlign16) {
    extern "C" {
        fn force_align_16(x: ForceAlign16);
    }

    // CHECK: start:
    // m68k-NEXT: call void @force_align_16({{.*}}byval([80 x i8]) align 16{{.*}})

    // wasm-NEXT: call void @force_align_16({{.*}}byval([80 x i8]) align 16{{.*}})

    // x86_64-linux-NEXT: call void @force_align_16({{.*}}byval([80 x i8]) align 16{{.*}})

    // x86_64-windows-NEXT: call void @force_align_16(
    // x86_64-windows-NOT: byval
    // x86_64-windows-SAME: align 16{{.*}})

    // i686-linux-NEXT: call void @force_align_16({{.*}}byval([80 x i8]) align 4{{.*}})

    // i686-windows-NEXT: call void @force_align_16(
    // i686-windows-NOT: byval
    // i686-windows-SAME: align 16{{.*}})
    force_align_16(x);
}
