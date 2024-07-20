// ignore-tidy-linelength
//@ revisions:aarch64 loongarch64 powerpc64 sparc64 x86_64
// FIXME: Add `-Cllvm-args=--lint-abort-on-error` after LLVM 19
//@ compile-flags: -O -C no-prepopulate-passes -C passes=lint

//@[aarch64] compile-flags: --target aarch64-unknown-linux-gnu
//@[aarch64] needs-llvm-components: arm
//@[loongarch64] compile-flags: --target loongarch64-unknown-linux-gnu
//@[loongarch64] needs-llvm-components: loongarch
//@[powerpc64] compile-flags: --target powerpc64-unknown-linux-gnu
//@[powerpc64] needs-llvm-components: powerpc
//@[sparc64] compile-flags: --target sparc64-unknown-linux-gnu
//@[sparc64] needs-llvm-components: sparc
//@[x86_64] compile-flags: --target x86_64-unknown-linux-gnu
//@[x86_64] needs-llvm-components: x86

// Tests that arguments with `PassMode::Cast` are handled correctly.

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

// This struct will be passed as a single `i64` or `i32`.
// This may be (if `i64)) larger than the Rust layout, which is just `{ i16, i16 }`.
#[repr(C)]
pub struct TwoU16s {
    a: u16,
    b: u16,
}

// This struct will be passed as `[2 x i64]`.
// This is larger than the Rust layout.
#[repr(C)]
pub struct FiveU16s {
    a: u16,
    b: u16,
    c: u16,
    d: u16,
    e: u16,
}

// This struct will be passed as `[2 x double]`.
// This is the same as the Rust layout.
#[repr(C)]
pub struct DoubleDouble {
    f: f64,
    g: f64,
}

// On loongarch, this struct will be passed as `{ double, float }`.
// This is smaller than the Rust layout, which has trailing padding (`{ f64, f32, <f32 padding> }`)
#[repr(C)]
pub struct DoubleFloat {
    f: f64,
    g: f32,
}

// On x86_64, this struct will be passed as `{ i64, i32 }`.
// The load and store instructions will access 16 bytes, so we should allocate 16 bytes.
#[repr(C)]
pub struct Three32s {
    a: u32,
    b: u32,
    c: u32,
}

// CHECK-LABEL: @receives_twou16s
// CHECK-AARCH64-SAME:     ([[ABI_TYPE:i64]] {{.*}}[[ABI_VALUE:%.+]])
// CHECK-LOONGARCH64-SAME: ([[ABI_TYPE:i64]] {{.*}}[[ABI_VALUE:%.+]])
// CHECK-POWERPC64-SAME:   ([[ABI_TYPE:i32]] {{.*}}[[ABI_VALUE:%.+]])
// CHECK-SPARC64-SAME:     ([[ABI_TYPE:i64]] {{.*}}[[ABI_VALUE:%.+]])
// CHECK-X86_64-SAME:      ([[ABI_TYPE:i32]] {{.*}}[[ABI_VALUE:%.+]])
#[no_mangle]
#[inline(never)]
pub extern "C" fn receives_twou16s(x: TwoU16s) {
    // CHECK-AARCH64:     [[ABI_ALLOCA:%.+]] = alloca [8 x i8], align [[ABI_ALIGN:8]]
    // CHECK-LOONGARCH64: [[ABI_ALLOCA:%.+]] = alloca [8 x i8], align [[ABI_ALIGN:8]]
    // CHECK-POWERPC64:   [[ABI_ALLOCA:%.+]] = alloca [4 x i8], align [[ABI_ALIGN:4]]
    // CHECK-SPARC64:     [[ABI_ALLOCA:%.+]] = alloca [8 x i8], align [[ABI_ALIGN:8]]
    // CHECK-X86_64:      [[ABI_ALLOCA:%.+]] = alloca [4 x i8], align [[ABI_ALIGN:4]]

    // CHECK: [[RUST_ALLOCA:%.+]] = alloca [4 x i8], align [[RUST_ALIGN:2]]

    // CHECK: store [[ABI_TYPE]] [[ABI_VALUE]], ptr [[ABI_ALLOCA]], align [[ABI_ALIGN]]

    // CHECK: call void @llvm.memcpy.{{.+}}(ptr align [[RUST_ALIGN]] [[RUST_ALLOCA]], ptr align [[ABI_ALIGN]] [[ABI_ALLOCA]], i64 4, i1 false)
}

// CHECK-LABEL: @returns_twou16s
// CHECK-POWERPC64-SAME: sret([4 x i8]) align [[RUST_ALIGN:2]] {{.*}}[[RET_PTR:%.*]])
#[no_mangle]
#[inline(never)]
pub extern "C" fn returns_twou16s() -> TwoU16s {
    // powerpc returns this struct via sret pointer, it doesn't use the cast ABI.
    // The other targets copy the cast ABI type to an alloca.

    // CHECK-AARCH64:     [[ABI_ALLOCA:%.+]] = alloca [8 x i8], align [[ABI_ALIGN:2]]
    // CHECK-LOONGARCH64: [[ABI_ALLOCA:%.+]] = alloca [8 x i8], align [[ABI_ALIGN:2]]
    // CHECK-SPARC64:     [[ABI_ALLOCA:%.+]] = alloca [8 x i8], align [[ABI_ALIGN:2]]
    // CHECK-X86_64:      [[ABI_ALLOCA:%.+]] = alloca [4 x i8], align [[ABI_ALIGN:2]]

    // CHECK-AARCH64:     [[ABI_VALUE:%.+]] = load [[ABI_TYPE:i64]], ptr [[ABI_ALLOCA]], align [[ABI_ALIGN]]
    // CHECK-LOONGARCH64: [[ABI_VALUE:%.+]] = load [[ABI_TYPE:i64]], ptr [[ABI_ALLOCA]], align [[ABI_ALIGN]]
    // CHECK-SPARC64:     [[ABI_VALUE:%.+]] = load [[ABI_TYPE:i64]], ptr [[ABI_ALLOCA]], align [[ABI_ALIGN]]
    // CHECK-X86_64:      [[ABI_VALUE:%.+]] = load [[ABI_TYPE:i32]], ptr [[ABI_ALLOCA]], align [[ABI_ALIGN]]

    // CHECK-AARCH64:     ret [[ABI_TYPE]] [[ABI_VALUE]]
    // CHECK-LOONGARCH64: ret [[ABI_TYPE]] [[ABI_VALUE]]
    // CHECK-SPARC64:     ret [[ABI_TYPE]] [[ABI_VALUE]]
    // CHECK-X86_64:      ret [[ABI_TYPE]] [[ABI_VALUE]]
    TwoU16s { a: 0, b: 1 }
}

// CHECK-LABEL: @receives_fiveu16s
// CHECK-AARCH64-SAME:     ([[ABI_TYPE:\[2 x i64\]]] {{.*}}[[ABI_VALUE:%.+]])
// CHECK-LOONGARCH64-SAME: ([[ABI_TYPE:\[2 x i64\]]] {{.*}}[[ABI_VALUE:%.+]])
// CHECK-POWERPC64-SAME:   ([[ABI_TYPE:\[2 x i64\]]] {{.*}}[[ABI_VALUE:%.+]])
// CHECK-SPARC64-SAME:     ([[ABI_TYPE:\[2 x i64\]]] {{.*}}[[ABI_VALUE:%.+]])
// CHECK-X86_64-SAME:      ([[ABI_TYPE:{ i64, i16 }]] {{.*}}[[ABI_VALUE:%.+]])
#[no_mangle]
#[inline(never)]
pub extern "C" fn receives_fiveu16s(x: FiveU16s) {
    // CHECK: [[ABI_ALLOCA:%.+]] = alloca [16 x i8], align [[ABI_ALIGN:8]]

    // CHECK: [[RUST_ALLOCA:%.+]] = alloca [10 x i8], align [[RUST_ALIGN:2]]

    // CHECK: store [[ABI_TYPE]] [[ABI_VALUE]], ptr [[ABI_ALLOCA]], align [[ABI_ALIGN]]

    // CHECK: call void @llvm.memcpy.{{.+}}(ptr align [[RUST_ALIGN]] [[RUST_ALLOCA]], ptr align [[ABI_ALIGN]] [[ABI_ALLOCA]], i64 10, i1 false)
}

// CHECK-LABEL: @returns_fiveu16s
// CHECK-POWERPC64-SAME: sret([10 x i8]) align [[RUST_ALIGN:2]] {{.*}}[[RET_PTR:%.*]])
#[no_mangle]
#[inline(never)]
pub extern "C" fn returns_fiveu16s() -> FiveU16s {
    // powerpc returns this struct via sret pointer, it doesn't use the cast ABI.
    // The other targets copy the cast ABI type to an alloca.

    // CHECK-AARCH64:     [[ABI_ALLOCA:%.+]] = alloca [16 x i8], align [[ABI_ALIGN:2]]
    // CHECK-LOONGARCH64: [[ABI_ALLOCA:%.+]] = alloca [16 x i8], align [[ABI_ALIGN:2]]
    // CHECK-SPARC64:     [[ABI_ALLOCA:%.+]] = alloca [16 x i8], align [[ABI_ALIGN:2]]
    // CHECK-X86_64:      [[ABI_ALLOCA:%.+]] = alloca [16 x i8], align [[ABI_ALIGN:2]]

    // CHECK-AARCH64:     [[ABI_VALUE:%.+]] = load [[ABI_TYPE:\[2 x i64\]]], ptr [[ABI_ALLOCA]], align [[ABI_ALIGN]]
    // CHECK-LOONGARCH64: [[ABI_VALUE:%.+]] = load [[ABI_TYPE:\[2 x i64\]]], ptr [[ABI_ALLOCA]], align [[ABI_ALIGN]]
    // CHECK-SPARC64:     [[ABI_VALUE:%.+]] = load [[ABI_TYPE:\[2 x i64\]]], ptr [[ABI_ALLOCA]], align [[ABI_ALIGN]]
    // CHECK-X86_64:      [[ABI_VALUE:%.+]] = load [[ABI_TYPE:{ i64, i16 }]], ptr [[ABI_ALLOCA]], align [[ABI_ALIGN]]

    // CHECK-AARCH64:     ret [[ABI_TYPE]] [[ABI_VALUE]]
    // CHECK-LOONGARCH64: ret [[ABI_TYPE]] [[ABI_VALUE]]
    // CHECK-SPARC64:     ret [[ABI_TYPE]] [[ABI_VALUE]]
    // CHECK-X86_64:      ret [[ABI_TYPE]] [[ABI_VALUE]]
    FiveU16s { a: 0, b: 1, c: 2, d: 3, e: 4 }
}

// CHECK-LABEL: @receives_doubledouble
// CHECK-AARCH64-SAME:     ([[ABI_TYPE:\[2 x double\]]] {{.*}}[[ABI_VALUE:%.+]])
// CHECK-LOONGARCH64-SAME: ([[ABI_TYPE:{ double, double }]] {{.*}}[[ABI_VALUE:%.+]])
// CHECK-POWERPC64-SAME:   ([[ABI_TYPE:\[2 x i64\]]] {{.*}}[[ABI_VALUE:%.+]])
// CHECK-SPARC64-SAME:     ([[ABI_TYPE:{ double, double }]] {{.*}}[[ABI_VALUE:%.+]])
// CHECK-X86_64-SAME:      ([[ABI_TYPE:{ double, double }]] {{.*}}[[ABI_VALUE:%.+]])
#[no_mangle]
#[inline(never)]
pub extern "C" fn receives_doubledouble(x: DoubleDouble) {
    // CHECK: [[ABI_ALLOCA:%.+]] = alloca [16 x i8], align [[ABI_ALIGN:8]]

    // CHECK: [[RUST_ALLOCA:%.+]] = alloca [16 x i8], align [[RUST_ALIGN:8]]

    // CHECK: store [[ABI_TYPE]] [[ABI_VALUE]], ptr [[ABI_ALLOCA]], align [[ABI_ALIGN]]

    // CHECK: call void @llvm.memcpy.{{.+}}(ptr align [[RUST_ALIGN]] [[RUST_ALLOCA]], ptr align [[ABI_ALIGN]] [[ABI_ALLOCA]], i64 16, i1 false)
}

// CHECK-LABEL: @returns_doubledouble
// CHECK-POWERPC64-SAME: sret([16 x i8]) align [[RUST_ALIGN:8]] {{.*}}[[RET_PTR:%.*]])
#[no_mangle]
#[inline(never)]
pub extern "C" fn returns_doubledouble() -> DoubleDouble {
    // powerpc returns this struct via sret pointer, it doesn't use the cast ABI.
    // The other targets copy the cast ABI type to an alloca.

    // CHECK-AARCH64:     [[ABI_ALLOCA:%.+]] = alloca [16 x i8], align [[ABI_ALIGN:8]]
    // CHECK-LOONGARCH64: [[ABI_ALLOCA:%.+]] = alloca [16 x i8], align [[ABI_ALIGN:8]]
    // CHECK-SPARC64:     [[ABI_ALLOCA:%.+]] = alloca [16 x i8], align [[ABI_ALIGN:8]]
    // CHECK-X86_64:      [[ABI_ALLOCA:%.+]] = alloca [16 x i8], align [[ABI_ALIGN:8]]

    // CHECK-AARCH64:     [[ABI_VALUE:%.+]] = load [[ABI_TYPE:\[2 x double\]]], ptr [[ABI_ALLOCA]], align [[ABI_ALIGN]]
    // CHECK-LOONGARCH64: [[ABI_VALUE:%.+]] = load [[ABI_TYPE:{ double, double }]], ptr [[ABI_ALLOCA]], align [[ABI_ALIGN]]
    // CHECK-SPARC64:     [[ABI_VALUE:%.+]] = load [[ABI_TYPE:{ double, double }]], ptr [[ABI_ALLOCA]], align [[ABI_ALIGN]]
    // CHECK-X86_64:      [[ABI_VALUE:%.+]] = load [[ABI_TYPE:{ double, double }]], ptr [[ABI_ALLOCA]], align [[ABI_ALIGN]]

    // CHECK-AARCH64:     ret [[ABI_TYPE]] [[ABI_VALUE]]
    // CHECK-LOONGARCH64: ret [[ABI_TYPE]] [[ABI_VALUE]]
    // CHECK-SPARC64:     ret [[ABI_TYPE]] [[ABI_VALUE]]
    // CHECK-X86_64:      ret [[ABI_TYPE]] [[ABI_VALUE]]
    DoubleDouble { f: 0., g: 1. }
}

// CHECK-LABEL: @receives_three32s
// CHECK-AARCH64-SAME:     ([[ABI_TYPE:\[2 x i64\]]] {{.*}}[[ABI_VALUE:%.+]])
// CHECK-LOONGARCH64-SAME: ([[ABI_TYPE:\[2 x i64\]]] {{.*}}[[ABI_VALUE:%.+]])
// CHECK-POWERPC64-SAME:   ([[ABI_TYPE:\[2 x i64\]]] {{.*}}[[ABI_VALUE:%.+]])
// CHECK-SPARC64-SAME:     ([[ABI_TYPE:\[2 x i64\]]] {{.*}}[[ABI_VALUE:%.+]])
// CHECK-X86_64-SAME:      ([[ABI_TYPE:{ i64, i32 }]] {{.*}}[[ABI_VALUE:%.+]])
#[no_mangle]
#[inline(never)]
pub extern "C" fn receives_three32s(x: Three32s) {
    // CHECK: [[ABI_ALLOCA:%.+]] = alloca [16 x i8], align [[ABI_ALIGN:8]]

    // CHECK: [[RUST_ALLOCA:%.+]] = alloca [12 x i8], align [[RUST_ALIGN:4]]

    // CHECK: store [[ABI_TYPE]] [[ABI_VALUE]], ptr [[ABI_ALLOCA]], align [[ABI_ALIGN]]

    // CHECK: call void @llvm.memcpy.{{.+}}(ptr align [[RUST_ALIGN]] [[RUST_ALLOCA]], ptr align [[ABI_ALIGN]] [[ABI_ALLOCA]], i64 12, i1 false)
}

// CHECK-LABEL: @returns_three32s
// CHECK-POWERPC64-SAME: sret([12 x i8]) align [[RUST_ALIGN:4]] {{.*}}[[RET_PTR:%.*]])
#[no_mangle]
#[inline(never)]
pub extern "C" fn returns_three32s() -> Three32s {
    // powerpc returns this struct via sret pointer, it doesn't use the cast ABI.
    // The other targets copy the cast ABI type to an alloca.

    // CHECK-AARCH64:     [[ABI_ALLOCA:%.+]] = alloca [16 x i8], align [[ABI_ALIGN:4]]
    // CHECK-LOONGARCH64: [[ABI_ALLOCA:%.+]] = alloca [16 x i8], align [[ABI_ALIGN:4]]
    // CHECK-SPARC64:     [[ABI_ALLOCA:%.+]] = alloca [16 x i8], align [[ABI_ALIGN:4]]
    // CHECK-X86_64:      [[ABI_ALLOCA:%.+]] = alloca [16 x i8], align [[ABI_ALIGN:4]]

    // CHECK-AARCH64:     [[ABI_VALUE:%.+]] = load [[ABI_TYPE:\[2 x i64\]]], ptr [[ABI_ALLOCA]], align [[ABI_ALIGN]]
    // CHECK-LOONGARCH64: [[ABI_VALUE:%.+]] = load [[ABI_TYPE:\[2 x i64\]]], ptr [[ABI_ALLOCA]], align [[ABI_ALIGN]]
    // CHECK-SPARC64:     [[ABI_VALUE:%.+]] = load [[ABI_TYPE:\[2 x i64\]]], ptr [[ABI_ALLOCA]], align [[ABI_ALIGN]]
    // CHECK-X86_64:      [[ABI_VALUE:%.+]] = load [[ABI_TYPE:{ i64, i32 }]], ptr [[ABI_ALLOCA]], align [[ABI_ALIGN]]

    // CHECK-AARCH64:     ret [[ABI_TYPE]] [[ABI_VALUE]]
    // CHECK-LOONGARCH64: ret [[ABI_TYPE]] [[ABI_VALUE]]
    // CHECK-SPARC64:     ret [[ABI_TYPE]] [[ABI_VALUE]]
    // CHECK-X86_64:      ret [[ABI_TYPE]] [[ABI_VALUE]]
    Three32s { a: 0, b: 0, c: 0 }
}

// These functions cause an ICE in sparc64 ABI code (https://github.com/rust-lang/rust/issues/122620)
#[cfg(not(target_arch = "sparc64"))]
// CHECK-AARCH64-LABEL:     @receives_doublefloat
// CHECK-LOONGARCH64-LABEL: @receives_doublefloat
// CHECK-POWERPC64-LABEL:   @receives_doublefloat
// CHECK-X86_64-LABEL:      @receives_doublefloat

// CHECK-AARCH64-SAME:     ([[ABI_TYPE:\[2 x i64\]]] {{.*}}[[ABI_VALUE:%.+]])
// CHECK-LOONGARCH64-SAME: ([[ABI_TYPE:{ double, float }]] {{.*}}[[ABI_VALUE:%.+]])
// CHECK-POWERPC64-SAME:   ([[ABI_TYPE:\[2 x i64\]]] {{.*}}[[ABI_VALUE:%.+]])
// CHECK-X86_64-SAME:      ([[ABI_TYPE:{ double, double }]] {{.*}}[[ABI_VALUE:%.+]])
#[no_mangle]
#[inline(never)]
pub extern "C" fn receives_doublefloat(x: DoubleFloat) {
    // CHECK-AARCH64:     [[ABI_ALLOCA:%.+]] = alloca [16 x i8], align [[ABI_ALIGN:8]]
    // CHECK-LOONGARCH64: [[ABI_ALLOCA:%.+]] = alloca [16 x i8], align [[ABI_ALIGN:8]]
    // CHECK-POWERPC64:   [[ABI_ALLOCA:%.+]] = alloca [16 x i8], align [[ABI_ALIGN:8]]
    // CHECK-X86_64:      [[ABI_ALLOCA:%.+]] = alloca [16 x i8], align [[ABI_ALIGN:8]]

    // CHECK-AARCH64:     [[RUST_ALLOCA:%.+]] = alloca [16 x i8], align [[RUST_ALIGN:8]]
    // CHECK-LOONGARCH64: [[RUST_ALLOCA:%.+]] = alloca [16 x i8], align [[RUST_ALIGN:8]]
    // CHECK-POWERPC64:   [[RUST_ALLOCA:%.+]] = alloca [16 x i8], align [[RUST_ALIGN:8]]
    // CHECK-X86_64:      [[RUST_ALLOCA:%.+]] = alloca [16 x i8], align [[RUST_ALIGN:8]]

    // CHECK-AARCH64:     store [[ABI_TYPE]] [[ABI_VALUE]], ptr [[ABI_ALLOCA]], align [[ABI_ALIGN]]
    // CHECK-LOONGARCH64: store [[ABI_TYPE]] [[ABI_VALUE]], ptr [[ABI_ALLOCA]], align [[ABI_ALIGN]]
    // CHECK-POWERPC64:   store [[ABI_TYPE]] [[ABI_VALUE]], ptr [[ABI_ALLOCA]], align [[ABI_ALIGN]]
    // CHECK-X86_64:      store [[ABI_TYPE]] [[ABI_VALUE]], ptr [[ABI_ALLOCA]], align [[ABI_ALIGN]]

    // CHECK-AARCH64:     call void @llvm.memcpy.{{.+}}(ptr align [[RUST_ALIGN]] [[RUST_ALLOCA]], ptr align [[ABI_ALIGN]] [[ABI_ALLOCA]], i64 16, i1 false)
    // CHECK-LOONGARCH64: call void @llvm.memcpy.{{.+}}(ptr align [[RUST_ALIGN]] [[RUST_ALLOCA]], ptr align [[ABI_ALIGN]] [[ABI_ALLOCA]], i64 12, i1 false)
    // CHECK-POWERPC64:   call void @llvm.memcpy.{{.+}}(ptr align [[RUST_ALIGN]] [[RUST_ALLOCA]], ptr align [[ABI_ALIGN]] [[ABI_ALLOCA]], i64 16, i1 false)
    // CHECK-X86_64:      call void @llvm.memcpy.{{.+}}(ptr align [[RUST_ALIGN]] [[RUST_ALLOCA]], ptr align [[ABI_ALIGN]] [[ABI_ALLOCA]], i64 16, i1 false)
}

#[cfg(not(target_arch = "sparc64"))]
// CHECK-AARCH64-LABEL:     @returns_doublefloat
// CHECK-LOONGARCH64-LABEL: @returns_doublefloat
// CHECK-POWERPC64-LABEL:   @returns_doublefloat
// CHECK-X86_64-LABEL:      @returns_doublefloat

// CHECK-POWERPC64-SAME: sret([16 x i8]) align [[RUST_ALIGN:8]] {{.*}}[[RET_PTR:%.*]])
#[no_mangle]
#[inline(never)]
pub extern "C" fn returns_doublefloat() -> DoubleFloat {
    // powerpc returns this struct via sret pointer, it doesn't use the cast ABI.
    // The other targets copy the cast ABI type to an alloca.

    // CHECK-AARCH64:     [[ABI_ALLOCA:%.+]] = alloca [16 x i8], align [[ABI_ALIGN:8]]
    // CHECK-LOONGARCH64: [[ABI_ALLOCA:%.+]] = alloca [16 x i8], align [[ABI_ALIGN:8]]
    // CHECK-X86_64:      [[ABI_ALLOCA:%.+]] = alloca [16 x i8], align [[ABI_ALIGN:8]]

    // CHECK-AARCH64:     [[ABI_VALUE:%.+]] = load [[ABI_TYPE:\[2 x i64\]]], ptr [[ABI_ALLOCA]], align [[ABI_ALIGN]]
    // CHECK-LOONGARCH64: [[ABI_VALUE:%.+]] = load [[ABI_TYPE:{ double, float }]], ptr [[ABI_ALLOCA]], align [[ABI_ALIGN]]
    // CHECK-X86_64:      [[ABI_VALUE:%.+]] = load [[ABI_TYPE:{ double, double }]], ptr [[ABI_ALLOCA]], align [[ABI_ALIGN]]

    // CHECK-AARCH64:     ret [[ABI_TYPE]] [[ABI_VALUE]]
    // CHECK-LOONGARCH64: ret [[ABI_TYPE]] [[ABI_VALUE]]
    // CHECK-X86_64:      ret [[ABI_TYPE]] [[ABI_VALUE]]
    DoubleFloat { f: 0., g: 0. }
}

// CHECK-LABEL: @call_twou16s
#[no_mangle]
pub fn call_twou16s() {
    // CHECK-AARCH64:     [[ABI_ALLOCA:%.+]] = alloca [8 x i8], align [[ABI_ALIGN:8]]
    // CHECK-LOONGARCH64: [[ABI_ALLOCA:%.+]] = alloca [8 x i8], align [[ABI_ALIGN:8]]
    // CHECK-POWERPC64:   [[ABI_ALLOCA:%.+]] = alloca [4 x i8], align [[ABI_ALIGN:4]]
    // CHECK-SPARC64:     [[ABI_ALLOCA:%.+]] = alloca [8 x i8], align [[ABI_ALIGN:8]]
    // CHECK-X86_64:      [[ABI_ALLOCA:%.+]] = alloca [4 x i8], align [[ABI_ALIGN:4]]

    // CHECK: [[RUST_ALLOCA:%.+]] = alloca [4 x i8], align [[RUST_ALIGN:2]]

    // CHECK: call void @llvm.memcpy.{{.+}}(ptr align [[ABI_ALIGN]] [[ABI_ALLOCA]], ptr align [[RUST_ALIGN]] [[RUST_ALLOCA]], i64 4, i1 false)

    // CHECK-AARCH64:     [[ABI_VALUE:%.+]] = load [[ABI_TYPE:i64]], ptr [[ABI_ALLOCA]], align [[ABI_ALIGN]]
    // CHECK-LOONGARCH64: [[ABI_VALUE:%.+]] = load [[ABI_TYPE:i64]], ptr [[ABI_ALLOCA]], align [[ABI_ALIGN]]
    // CHECK-POWERPC64:   [[ABI_VALUE:%.+]] = load [[ABI_TYPE:i32]], ptr [[ABI_ALLOCA]], align [[ABI_ALIGN]]
    // CHECK-SPARC64:     [[ABI_VALUE:%.+]] = load [[ABI_TYPE:i64]], ptr [[ABI_ALLOCA]], align [[ABI_ALIGN]]
    // CHECK-X86_64:      [[ABI_VALUE:%.+]] = load [[ABI_TYPE:i32]], ptr [[ABI_ALLOCA]], align [[ABI_ALIGN]]

    // CHECK: call void @receives_twou16s([[ABI_TYPE]] [[ABI_VALUE]])
    let x = TwoU16s { a: 1, b: 2 };
    receives_twou16s(x);
}

// CHECK-LABEL: @return_twou16s
#[no_mangle]
pub fn return_twou16s() -> TwoU16s {
    // powerpc returns this struct via sret pointer, it doesn't use the cast ABI.

    // CHECK-POWERPC64: [[RETVAL:%.+]] = alloca [4 x i8], align 2
    // CHECK-POWERPC64: call void @returns_twou16s(ptr {{.+}} [[RETVAL]])

    // The other targets copy the cast ABI type to an alloca.

    // CHECK-AARCH64:     [[ABI_ALLOCA:%.+]] = alloca [8 x i8], align [[ABI_ALIGN:8]]
    // CHECK-LOONGARCH64: [[ABI_ALLOCA:%.+]] = alloca [8 x i8], align [[ABI_ALIGN:8]]
    // CHECK-SPARC64:     [[ABI_ALLOCA:%.+]] = alloca [8 x i8], align [[ABI_ALIGN:8]]
    // CHECK-X86_64:      [[ABI_ALLOCA:%.+]] = alloca [4 x i8], align [[ABI_ALIGN:4]]

    // CHECK-AARCH64:     [[RUST_ALLOCA:%.+]] = alloca [4 x i8], align [[RUST_ALIGN:2]]
    // CHECK-LOONGARCH64: [[RUST_ALLOCA:%.+]] = alloca [4 x i8], align [[RUST_ALIGN:2]]
    // CHECK-SPARC64:     [[RUST_ALLOCA:%.+]] = alloca [4 x i8], align [[RUST_ALIGN:2]]
    // CHECK-X86_64:      [[RUST_ALLOCA:%.+]] = alloca [4 x i8], align [[RUST_ALIGN:2]]

    // CHECK-AARCH64:     [[ABI_VALUE:%.+]] = call [[ABI_TYPE:i64]] @returns_twou16s()
    // CHECK-LOONGARCH64: [[ABI_VALUE:%.+]] = call [[ABI_TYPE:i64]] @returns_twou16s()
    // CHECK-SPARC64:     [[ABI_VALUE:%.+]] = call [[ABI_TYPE:i64]] @returns_twou16s()
    // CHECK-X86_64:      [[ABI_VALUE:%.+]] = call [[ABI_TYPE:i32]] @returns_twou16s()

    // CHECK-AARCH64:     store [[ABI_TYPE]] [[ABI_VALUE]], ptr [[ABI_ALLOCA]], align [[ABI_ALIGN]]
    // CHECK-LOONGARCH64: store [[ABI_TYPE]] [[ABI_VALUE]], ptr [[ABI_ALLOCA]], align [[ABI_ALIGN]]
    // CHECK-SPARC64:     store [[ABI_TYPE]] [[ABI_VALUE]], ptr [[ABI_ALLOCA]], align [[ABI_ALIGN]]
    // CHECK-X86_64:      store [[ABI_TYPE]] [[ABI_VALUE]], ptr [[ABI_ALLOCA]], align [[ABI_ALIGN]]

    // CHECK-AARCH64:     call void @llvm.memcpy.{{.+}}(ptr align [[RUST_ALIGN]] [[RUST_ALLOCA]], ptr align [[ABI_ALIGN]] [[ABI_ALLOCA]], i64 4, i1 false)
    // CHECK-LOONGARCH64: call void @llvm.memcpy.{{.+}}(ptr align [[RUST_ALIGN]] [[RUST_ALLOCA]], ptr align [[ABI_ALIGN]] [[ABI_ALLOCA]], i64 4, i1 false)
    // CHECK-SPARC64:     call void @llvm.memcpy.{{.+}}(ptr align [[RUST_ALIGN]] [[RUST_ALLOCA]], ptr align [[ABI_ALIGN]] [[ABI_ALLOCA]], i64 4, i1 false)
    // CHECK-X86_64:      call void @llvm.memcpy.{{.+}}(ptr align [[RUST_ALIGN]] [[RUST_ALLOCA]], ptr align [[ABI_ALIGN]] [[ABI_ALLOCA]], i64 4, i1 false)
    returns_twou16s()
}

// CHECK-LABEL: @call_fiveu16s
#[no_mangle]
pub fn call_fiveu16s() {
    // CHECK: [[ABI_ALLOCA:%.+]] = alloca [16 x i8], align [[ABI_ALIGN:8]]

    // CHECK: [[RUST_ALLOCA:%.+]] = alloca [10 x i8], align 2

    // CHECK: call void @llvm.memcpy.{{.+}}(ptr align [[ABI_ALIGN]] [[ABI_ALLOCA]], ptr align [[RUST_ALIGN]] [[RUST_ALLOCA]], i64 10, i1 false)

    // CHECK-AARCH64:     [[ABI_VALUE:%.+]] = load [[ABI_TYPE:\[2 x i64\]]], ptr [[ABI_ALLOCA]], align [[ABI_ALIGN]]
    // CHECK-LOONGARCH64: [[ABI_VALUE:%.+]] = load [[ABI_TYPE:\[2 x i64\]]], ptr [[ABI_ALLOCA]], align [[ABI_ALIGN]]
    // CHECK-POWERPC64:   [[ABI_VALUE:%.+]] = load [[ABI_TYPE:\[2 x i64\]]], ptr [[ABI_ALLOCA]], align [[ABI_ALIGN]]
    // CHECK-SPARC64:     [[ABI_VALUE:%.+]] = load [[ABI_TYPE:\[2 x i64\]]], ptr [[ABI_ALLOCA]], align [[ABI_ALIGN]]
    // CHECK-X86_64:      [[ABI_VALUE:%.+]] = load [[ABI_TYPE:{ i64, i16 }]], ptr [[ABI_ALLOCA]], align [[ABI_ALIGN]]

    // CHECK: call void @receives_fiveu16s([[ABI_TYPE]] [[ABI_VALUE]])
    let x = FiveU16s { a: 1, b: 2, c: 3, d: 4, e: 5 };
    receives_fiveu16s(x);
}

// CHECK-LABEL: @return_fiveu16s
// CHECK-SAME: (ptr {{.+}} sret([10 x i8]) align [[RUST_ALIGN:2]] dereferenceable(10) [[RET_PTR:%.+]])
#[no_mangle]
pub fn return_fiveu16s() -> FiveU16s {
    // powerpc returns this struct via sret pointer, it doesn't use the cast ABI.

    // CHECK-POWERPC64: call void @returns_fiveu16s(ptr {{.+}} [[RET_PTR]])

    // The other targets copy the cast ABI type to the sret pointer.

    // CHECK-AARCH64:     [[ABI_ALLOCA:%.+]] = alloca [16 x i8], align [[ABI_ALIGN:8]]
    // CHECK-LOONGARCH64: [[ABI_ALLOCA:%.+]] = alloca [16 x i8], align [[ABI_ALIGN:8]]
    // CHECK-SPARC64:     [[ABI_ALLOCA:%.+]] = alloca [16 x i8], align [[ABI_ALIGN:8]]
    // CHECK-X86_64:      [[ABI_ALLOCA:%.+]] = alloca [16 x i8], align [[ABI_ALIGN:8]]

    // CHECK-AARCH64:     [[ABI_VALUE:%.+]] = call [[ABI_TYPE:\[2 x i64\]]] @returns_fiveu16s()
    // CHECK-LOONGARCH64: [[ABI_VALUE:%.+]] = call [[ABI_TYPE:\[2 x i64\]]] @returns_fiveu16s()
    // CHECK-SPARC64:     [[ABI_VALUE:%.+]] = call [[ABI_TYPE:\[2 x i64\]]] @returns_fiveu16s()
    // CHECK-X86_64:      [[ABI_VALUE:%.+]] = call [[ABI_TYPE:{ i64, i16 }]] @returns_fiveu16s()

    // CHECK-AARCH64:     store [[ABI_TYPE]] [[ABI_VALUE]], ptr [[ABI_ALLOCA]], align [[ABI_ALIGN]]
    // CHECK-LOONGARCH64: store [[ABI_TYPE]] [[ABI_VALUE]], ptr [[ABI_ALLOCA]], align [[ABI_ALIGN]]
    // CHECK-SPARC64:     store [[ABI_TYPE]] [[ABI_VALUE]], ptr [[ABI_ALLOCA]], align [[ABI_ALIGN]]
    // CHECK-X86_64:      store [[ABI_TYPE]] [[ABI_VALUE]], ptr [[ABI_ALLOCA]], align [[ABI_ALIGN]]

    // CHECK-AARCH64:     call void @llvm.memcpy.{{.+}}(ptr align [[RUST_ALIGN]] [[RET_PTR]], ptr align [[ABI_ALIGN]] [[ABI_ALLOCA]], i64 10, i1 false)
    // CHECK-LOONGARCH64: call void @llvm.memcpy.{{.+}}(ptr align [[RUST_ALIGN]] [[RET_PTR]], ptr align [[ABI_ALIGN]] [[ABI_ALLOCA]], i64 10, i1 false)
    // CHECK-SPARC64:     call void @llvm.memcpy.{{.+}}(ptr align [[RUST_ALIGN]] [[RET_PTR]], ptr align [[ABI_ALIGN]] [[ABI_ALLOCA]], i64 10, i1 false)
    // CHECK-X86_64:      call void @llvm.memcpy.{{.+}}(ptr align [[RUST_ALIGN]] [[RET_PTR]], ptr align [[ABI_ALIGN]] [[ABI_ALLOCA]], i64 10, i1 false)
    returns_fiveu16s()
}

// CHECK-LABEL: @call_doubledouble
#[no_mangle]
pub fn call_doubledouble() {
    // CHECK: [[ABI_ALLOCA:%.+]] = alloca [16 x i8], align [[ABI_ALIGN:8]]

    // CHECK: [[RUST_ALLOCA:%.+]] = alloca [16 x i8], align [[RUST_ALIGN:8]]

    // CHECK: call void @llvm.memcpy.{{.+}}(ptr align [[ABI_ALIGN]] [[ABI_ALLOCA]], ptr align [[RUST_ALIGN]] [[RUST_ALLOCA]], i64 16, i1 false)

    // CHECK-AARCH64:     [[ABI_VALUE:%.+]] = load [[ABI_TYPE:\[2 x double\]]], ptr [[ABI_ALLOCA]], align [[ABI_ALIGN]]
    // CHECK-LOONGARCH64: [[ABI_VALUE:%.+]] = load [[ABI_TYPE:{ double, double }]], ptr [[ABI_ALLOCA]], align [[ABI_ALIGN]]
    // CHECK-POWERPC64:   [[ABI_VALUE:%.+]] = load [[ABI_TYPE:\[2 x i64\]]], ptr [[ABI_ALLOCA]], align [[ABI_ALIGN]]
    // CHECK-SPARC64:     [[ABI_VALUE:%.+]] = load [[ABI_TYPE:{ double, double }]], ptr [[ABI_ALLOCA]], align [[ABI_ALIGN]]
    // CHECK-X86_64:      [[ABI_VALUE:%.+]] = load [[ABI_TYPE:{ double, double }]], ptr [[ABI_ALLOCA]], align [[ABI_ALIGN]]

    // CHECK: call void @receives_doubledouble([[ABI_TYPE]] [[ABI_VALUE]])
    let x = DoubleDouble { f: 1., g: 2. };
    receives_doubledouble(x);
}

// CHECK-LABEL: @return_doubledouble
#[no_mangle]
pub fn return_doubledouble() -> DoubleDouble {
    // powerpc returns this struct via sret pointer, it doesn't use the cast ABI.

    // CHECK-POWERPC64: [[RETVAL:%.+]] = alloca [16 x i8], align 8
    // CHECK-POWERPC64: call void @returns_doubledouble(ptr {{.+}} [[RETVAL]])

    // The other targets copy the cast ABI type to an alloca.

    // CHECK-AARCH64:     [[ABI_ALLOCA:%.+]] = alloca [16 x i8], align [[ABI_ALIGN:8]]
    // CHECK-LOONGARCH64: [[ABI_ALLOCA:%.+]] = alloca [16 x i8], align [[ABI_ALIGN:8]]
    // CHECK-SPARC64:     [[ABI_ALLOCA:%.+]] = alloca [16 x i8], align [[ABI_ALIGN:8]]
    // CHECK-X86_64:      [[ABI_ALLOCA:%.+]] = alloca [16 x i8], align [[ABI_ALIGN:8]]

    // CHECK-AARCH64:     [[RUST_ALLOCA:%.+]] = alloca [16 x i8], align [[RUST_ALIGN:8]]
    // CHECK-LOONGARCH64: [[RUST_ALLOCA:%.+]] = alloca [16 x i8], align [[RUST_ALIGN:8]]
    // CHECK-SPARC64:     [[RUST_ALLOCA:%.+]] = alloca [16 x i8], align [[RUST_ALIGN:8]]
    // CHECK-X86_64:      [[RUST_ALLOCA:%.+]] = alloca [16 x i8], align [[RUST_ALIGN:8]]

    // CHECK-AARCH64:     [[ABI_VALUE:%.+]] = call [[ABI_TYPE:\[2 x double\]]] @returns_doubledouble()
    // CHECK-LOONGARCH64: [[ABI_VALUE:%.+]] = call [[ABI_TYPE:{ double, double }]] @returns_doubledouble()
    // CHECK-SPARC64:     [[ABI_VALUE:%.+]] = call [[ABI_TYPE:{ double, double }]] @returns_doubledouble()
    // CHECK-X86_64:      [[ABI_VALUE:%.+]] = call [[ABI_TYPE:{ double, double }]] @returns_doubledouble()

    // CHECK-AARCH64:     store [[ABI_TYPE]] [[ABI_VALUE]], ptr [[ABI_ALLOCA]], align [[ABI_ALIGN]]
    // CHECK-LOONGARCH64: store [[ABI_TYPE]] [[ABI_VALUE]], ptr [[ABI_ALLOCA]], align [[ABI_ALIGN]]
    // CHECK-SPARC64:     store [[ABI_TYPE]] [[ABI_VALUE]], ptr [[ABI_ALLOCA]], align [[ABI_ALIGN]]
    // CHECK-X86_64:      store [[ABI_TYPE]] [[ABI_VALUE]], ptr [[ABI_ALLOCA]], align [[ABI_ALIGN]]

    // CHECK-AARCH64:     call void @llvm.memcpy.{{.+}}(ptr align [[RUST_ALIGN]] [[RUST_ALLOCA]], ptr align [[ABI_ALIGN]] [[ABI_ALLOCA]], i64 16, i1 false)
    // CHECK-LOONGARCH64: call void @llvm.memcpy.{{.+}}(ptr align [[RUST_ALIGN]] [[RUST_ALLOCA]], ptr align [[ABI_ALIGN]] [[ABI_ALLOCA]], i64 16, i1 false)
    // CHECK-SPARC64:     call void @llvm.memcpy.{{.+}}(ptr align [[RUST_ALIGN]] [[RUST_ALLOCA]], ptr align [[ABI_ALIGN]] [[ABI_ALLOCA]], i64 16, i1 false)
    // CHECK-X86_64:      call void @llvm.memcpy.{{.+}}(ptr align [[RUST_ALIGN]] [[RUST_ALLOCA]], ptr align [[ABI_ALIGN]] [[ABI_ALLOCA]], i64 16, i1 false)
    returns_doubledouble()
}

// This test causes an ICE in sparc64 ABI code (https://github.com/rust-lang/rust/issues/122620)
#[cfg(not(target_arch = "sparc64"))]
// CHECK-AARCH64-LABEL:     @call_doublefloat
// CHECK-LOONGARCH64-LABEL: @call_doublefloat
// CHECK-POWERPC64-LABEL:   @call_doublefloat
// CHECK-X86_64-LABEL:      @call_doublefloat
#[no_mangle]
pub fn call_doublefloat() {
    // CHECK-AARCH64:     [[ABI_ALLOCA:%.+]] = alloca [16 x i8], align [[ABI_ALIGN:8]]
    // CHECK-LOONGARCH64: [[ABI_ALLOCA:%.+]] = alloca [16 x i8], align [[ABI_ALIGN:8]]
    // CHECK-POWERPC64:   [[ABI_ALLOCA:%.+]] = alloca [16 x i8], align [[ABI_ALIGN:8]]
    // CHECK-X86_64:      [[ABI_ALLOCA:%.+]] = alloca [16 x i8], align [[ABI_ALIGN:8]]

    // CHECK-AARCH64:     [[RUST_ALLOCA:%.+]] = alloca [16 x i8], align [[RUST_ALIGN:8]]
    // CHECK-LOONGARCH64: [[RUST_ALLOCA:%.+]] = alloca [16 x i8], align [[RUST_ALIGN:8]]
    // CHECK-POWERPC64:   [[RUST_ALLOCA:%.+]] = alloca [16 x i8], align [[RUST_ALIGN:8]]
    // CHECK-X86_64:      [[RUST_ALLOCA:%.+]] = alloca [16 x i8], align [[RUST_ALIGN:8]]

    // CHECK-AARCH64:     call void @llvm.memcpy.{{.+}}(ptr align [[ABI_ALIGN]] [[ABI_ALLOCA]], ptr align [[RUST_ALIGN]] [[RUST_ALLOCA]], i64 16, i1 false)
    // CHECK-LOONGARCH64: call void @llvm.memcpy.{{.+}}(ptr align [[ABI_ALIGN]] [[ABI_ALLOCA]], ptr align [[RUST_ALIGN]] [[RUST_ALLOCA]], i64 12, i1 false)
    // CHECK-POWERPC64:   call void @llvm.memcpy.{{.+}}(ptr align [[ABI_ALIGN]] [[ABI_ALLOCA]], ptr align [[RUST_ALIGN]] [[RUST_ALLOCA]], i64 16, i1 false)
    // CHECK-X86_64:      call void @llvm.memcpy.{{.+}}(ptr align [[ABI_ALIGN]] [[ABI_ALLOCA]], ptr align [[RUST_ALIGN]] [[RUST_ALLOCA]], i64 16, i1 false)

    // CHECK-AARCH64:     [[ABI_VALUE:%.+]] = load [[ABI_TYPE:\[2 x i64\]]], ptr [[ABI_ALLOCA]], align [[ABI_ALIGN]]
    // CHECK-LOONGARCH64: [[ABI_VALUE:%.+]] = load [[ABI_TYPE:{ double, float }]], ptr [[ABI_ALLOCA]], align [[ABI_ALIGN]]
    // CHECK-POWERPC64:   [[ABI_VALUE:%.+]] = load [[ABI_TYPE:\[2 x i64\]]], ptr [[ABI_ALLOCA]], align [[ABI_ALIGN]]
    // CHECK-X86_64:      [[ABI_VALUE:%.+]] = load [[ABI_TYPE:{ double, double }]], ptr [[ABI_ALLOCA]], align [[ABI_ALIGN]]

    // CHECK-AARCH64:     call void @receives_doublefloat([[ABI_TYPE]] {{(inreg )?}}[[ABI_VALUE]])
    // CHECK-LOONGARCH64: call void @receives_doublefloat([[ABI_TYPE]] {{(inreg )?}}[[ABI_VALUE]])
    // CHECK-POWERPC64:   call void @receives_doublefloat([[ABI_TYPE]] {{(inreg )?}}[[ABI_VALUE]])
    // CHECK-X86_64:      call void @receives_doublefloat([[ABI_TYPE]] {{(inreg )?}}[[ABI_VALUE]])
    let x = DoubleFloat { f: 1., g: 2. };
    receives_doublefloat(x);
}

// This test causes an ICE in sparc64 ABI code (https://github.com/rust-lang/rust/issues/122620)
#[cfg(not(target_arch = "sparc64"))]
// CHECK-AARCH64-LABEL:     @return_doublefloat
// CHECK-LOONGARCH64-LABEL: @return_doublefloat
// CHECK-POWERPC64-LABEL:   @return_doublefloat
// CHECK-X86_64-LABEL:      @return_doublefloat
#[no_mangle]
pub fn return_doublefloat() -> DoubleFloat {
    // powerpc returns this struct via sret pointer, it doesn't use the cast ABI.

    // CHECK-POWERPC64: [[RETVAL:%.+]] = alloca [16 x i8], align 8
    // CHECK-POWERPC64: call void @returns_doublefloat(ptr {{.+}} [[RETVAL]])

    // The other targets copy the cast ABI type to an alloca.

    // CHECK-AARCH64:     [[ABI_ALLOCA:%.+]] = alloca [16 x i8], align [[ABI_ALIGN:8]]
    // CHECK-LOONGARCH64: [[ABI_ALLOCA:%.+]] = alloca [16 x i8], align [[ABI_ALIGN:8]]
    // CHECK-X86_64:      [[ABI_ALLOCA:%.+]] = alloca [16 x i8], align [[ABI_ALIGN:8]]

    // CHECK-AARCH64:     [[RUST_ALLOCA:%.+]] = alloca [16 x i8], align [[RUST_ALIGN:8]]
    // CHECK-LOONGARCH64: [[RUST_ALLOCA:%.+]] = alloca [16 x i8], align [[RUST_ALIGN:8]]
    // CHECK-X86_64:      [[RUST_ALLOCA:%.+]] = alloca [16 x i8], align [[RUST_ALIGN:8]]

    // CHECK-AARCH64:     [[ABI_VALUE:%.+]] = call [[ABI_TYPE:\[2 x i64\]]] @returns_doublefloat()
    // CHECK-LOONGARCH64: [[ABI_VALUE:%.+]] = call [[ABI_TYPE:{ double, float }]] @returns_doublefloat()
    // CHECK-X86_64:      [[ABI_VALUE:%.+]] = call [[ABI_TYPE:{ double, double }]] @returns_doublefloat()

    // CHECK-AARCH64:     store [[ABI_TYPE]] [[ABI_VALUE]], ptr [[ABI_ALLOCA]], align [[ABI_ALIGN]]
    // CHECK-LOONGARCH64: store [[ABI_TYPE]] [[ABI_VALUE]], ptr [[ABI_ALLOCA]], align [[ABI_ALIGN]]
    // CHECK-X86_64:      store [[ABI_TYPE]] [[ABI_VALUE]], ptr [[ABI_ALLOCA]], align [[ABI_ALIGN]]

    // CHECK-AARCH64:     call void @llvm.memcpy.{{.+}}(ptr align [[RUST_ALIGN]] [[RUST_ALLOCA]], ptr align [[ABI_ALIGN]] [[ABI_ALLOCA]], i64 16, i1 false)
    // CHECK-LOONGARCH64: call void @llvm.memcpy.{{.+}}(ptr align [[RUST_ALIGN]] [[RUST_ALLOCA]], ptr align [[ABI_ALIGN]] [[ABI_ALLOCA]], i64 12, i1 false)
    // CHECK-X86_64:      call void @llvm.memcpy.{{.+}}(ptr align [[RUST_ALIGN]] [[RUST_ALLOCA]], ptr align [[ABI_ALIGN]] [[ABI_ALLOCA]], i64 16, i1 false)
    returns_doublefloat()
}

// CHECK-LABEL: @call_three32s
#[no_mangle]
pub fn call_three32s() {
    // CHECK: [[ABI_ALLOCA:%.+]] = alloca [16 x i8], align [[ABI_ALIGN:8]]
    // CHECK: [[RUST_ALLOCA:%.+]] = alloca [12 x i8], align [[RUST_ALIGN:4]]
    // CHECK: call void @llvm.memcpy.{{.+}}(ptr align [[ABI_ALIGN]] [[ABI_ALLOCA]], ptr align [[RUST_ALIGN]] [[RUST_ALLOCA]], i64 12, i1 false)

    // CHECK-AARCH64:     [[ABI_VALUE:%.+]] = load [[ABI_TYPE:\[2 x i64\]]], ptr [[ABI_ALLOCA]], align [[ABI_ALIGN]]
    // CHECK-LOONGARCH64: [[ABI_VALUE:%.+]] = load [[ABI_TYPE:\[2 x i64\]]], ptr [[ABI_ALLOCA]], align [[ABI_ALIGN]]
    // CHECK-POWERPC64:   [[ABI_VALUE:%.+]] = load [[ABI_TYPE:\[2 x i64\]]], ptr [[ABI_ALLOCA]], align [[ABI_ALIGN]]
    // CHECK-SPARC64:     [[ABI_VALUE:%.+]] = load [[ABI_TYPE:\[2 x i64\]]], ptr [[ABI_ALLOCA]], align [[ABI_ALIGN]]
    // CHECK-X86_64:      [[ABI_VALUE:%.+]] = load [[ABI_TYPE:{ i64, i32 }]], ptr [[ABI_ALLOCA]], align [[ABI_ALIGN]]

    // CHECK: call void @receives_three32s([[ABI_TYPE]] [[ABI_VALUE]])
    let x = Three32s { a: 1, b: 2, c: 3 };
    receives_three32s(x);
}

// Regression test for #75839
// CHECK-LABEL: @return_three32s(
// CHECK-SAME: sret([12 x i8]) align [[RUST_ALIGN:4]] {{.*}}[[RUST_RETVAL:%.*]])
#[no_mangle]
pub fn return_three32s() -> Three32s {
    // powerpc returns this struct via sret pointer, it doesn't use the cast ABI.

    // CHECK-POWERPC64: call void @returns_three32s(ptr {{.+}} [[RUST_RETVAL]])

    // CHECK-AARCH64:     [[ABI_ALLOCA:%.+]] = alloca [16 x i8], align [[ABI_ALIGN:8]]
    // CHECK-LOONGARCH64: [[ABI_ALLOCA:%.+]] = alloca [16 x i8], align [[ABI_ALIGN:8]]
    // CHECK-SPARC64:     [[ABI_ALLOCA:%.+]] = alloca [16 x i8], align [[ABI_ALIGN:8]]
    // CHECK-X86_64:      [[ABI_ALLOCA:%.+]] = alloca [16 x i8], align [[ABI_ALIGN:8]]

    // CHECK-AARCH64:     [[ABI_VALUE:%.+]] = call [[ABI_TYPE:\[2 x i64\]]] @returns_three32s()
    // CHECK-LOONGARCH64: [[ABI_VALUE:%.+]] = call [[ABI_TYPE:\[2 x i64\]]] @returns_three32s()
    // CHECK-SPARC64:     [[ABI_VALUE:%.+]] = call [[ABI_TYPE:\[2 x i64\]]] @returns_three32s()
    // CHECK-X86_64:      [[ABI_VALUE:%.+]] = call [[ABI_TYPE:{ i64, i32 }]] @returns_three32s()

    // CHECK-AARCH64:     store [[ABI_TYPE]] [[ABI_VALUE]], ptr [[ABI_ALLOCA]], align [[ABI_ALIGN]]
    // CHECK-LOONGARCH64: store [[ABI_TYPE]] [[ABI_VALUE]], ptr [[ABI_ALLOCA]], align [[ABI_ALIGN]]
    // CHECK-SPARC64:     store [[ABI_TYPE]] [[ABI_VALUE]], ptr [[ABI_ALLOCA]], align [[ABI_ALIGN]]
    // CHECK-X86_64:      store [[ABI_TYPE]] [[ABI_VALUE]], ptr [[ABI_ALLOCA]], align [[ABI_ALIGN]]

    // CHECK-AARCH64:     call void @llvm.memcpy.{{.+}}(ptr align [[RUST_ALIGN]] [[RUST_RETVAL]], ptr align [[ABI_ALIGN]] [[ABI_ALLOCA]], i64 12, i1 false)
    // CHECK-LOONGARCH64: call void @llvm.memcpy.{{.+}}(ptr align [[RUST_ALIGN]] [[RUST_RETVAL]], ptr align [[ABI_ALIGN]] [[ABI_ALLOCA]], i64 12, i1 false)
    // CHECK-SPARC64:     call void @llvm.memcpy.{{.+}}(ptr align [[RUST_ALIGN]] [[RUST_RETVAL]], ptr align [[ABI_ALIGN]] [[ABI_ALLOCA]], i64 12, i1 false)
    // CHECK-X86_64:      call void @llvm.memcpy.{{.+}}(ptr align [[RUST_ALIGN]] [[RUST_RETVAL]], ptr align [[ABI_ALIGN]] [[ABI_ALLOCA]], i64 12, i1 false)
    returns_three32s()
}
