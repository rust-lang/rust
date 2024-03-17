// ignore-tidy-linelength
//@ revisions:aarch64 loongarch64 powerpc64 sparc64
//@ compile-flags: -O -C no-prepopulate-passes

//@[aarch64] compile-flags: --target aarch64-unknown-linux-gnu
//@[aarch64] needs-llvm-components: arm
//@[loongarch64] compile-flags: --target loongarch64-unknown-linux-gnu
//@[loongarch64] needs-llvm-components: loongarch
//@[powerpc64] compile-flags: --target powerpc64-unknown-linux-gnu
//@[powerpc64] needs-llvm-components: powerpc
//@[sparc64] compile-flags: --target sparc64-unknown-linux-gnu
//@[sparc64] needs-llvm-components: sparc

// Tests that arguments with `PassMode::Cast` are handled correctly.

#![feature(no_core, lang_items)]
#![crate_type = "lib"]
#![no_std]
#![no_core]

#[lang="sized"] trait Sized { }
#[lang="freeze"] trait Freeze { }
#[lang="copy"] trait Copy { }

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

extern "C" {
    fn receives_twou16s(x: TwoU16s);
    fn returns_twou16s() -> TwoU16s;

    fn receives_fiveu16s(x: FiveU16s);
    fn returns_fiveu16s() -> FiveU16s;

    fn receives_doubledouble(x: DoubleDouble);
    fn returns_doubledouble() -> DoubleDouble;

    fn receives_doublefloat(x: DoubleFloat);
    fn returns_doublefloat() -> DoubleFloat;
}

// CHECK-LABEL: @call_twou16s
#[no_mangle]
pub unsafe fn call_twou16s() {
    // aarch64:     [[ABI_ALLOCA:%.+]] = alloca [[ABI_TYPE:i64]], align [[ABI_ALIGN:8]]
    // loongarch64: [[ABI_ALLOCA:%.+]] = alloca [[ABI_TYPE:i64]], align [[ABI_ALIGN:8]]
    // powerpc64:   [[ABI_ALLOCA:%.+]] = alloca [[ABI_TYPE:i32]], align [[ABI_ALIGN:4]]
    // sparc64:     [[ABI_ALLOCA:%.+]] = alloca [[ABI_TYPE:i64]], align [[ABI_ALIGN:8]]

    // CHECK: [[RUST_ALLOCA:%.+]] = alloca %TwoU16s, align [[RUST_ALIGN:2]]

    // CHECK: call void @llvm.memcpy.{{.+}}(ptr align [[ABI_ALIGN]] [[ABI_ALLOCA]], ptr align [[RUST_ALIGN]] [[RUST_ALLOCA]], i64 4, i1 false)
    // CHECK: [[ABI_VALUE:%.+]] = load [[ABI_TYPE]], ptr [[ABI_ALLOCA]], align [[ABI_ALIGN]]
    // CHECK: call void @receives_twou16s([[ABI_TYPE]] [[ABI_VALUE]])
    let x = TwoU16s { a: 1, b: 2 };
    receives_twou16s(x);
}

// CHECK-LABEL: @return_twou16s
#[no_mangle]
pub unsafe fn return_twou16s() -> TwoU16s {
    // powerpc returns this struct via sret pointer, it doesn't use the cast ABI.

    // powerpc64: [[RETVAL:%.+]] = alloca %TwoU16s, align 2
    // powerpc64: call void @returns_twou16s(ptr {{.+}} [[RETVAL]])


    // The other targets copy the cast ABI type to an alloca.

    // aarch64:     [[ABI_ALLOCA:%.+]] = alloca [[ABI_TYPE:i64]], align [[ABI_ALIGN:8]]
    // loongarch64: [[ABI_ALLOCA:%.+]] = alloca [[ABI_TYPE:i64]], align [[ABI_ALIGN:8]]
    // sparc64:     [[ABI_ALLOCA:%.+]] = alloca [[ABI_TYPE:i64]], align [[ABI_ALIGN:8]]

    // aarch64:     [[RUST_ALLOCA:%.+]] = alloca %TwoU16s, align [[RUST_ALIGN:2]]
    // loongarch64: [[RUST_ALLOCA:%.+]] = alloca %TwoU16s, align [[RUST_ALIGN:2]]
    // sparc64:     [[RUST_ALLOCA:%.+]] = alloca %TwoU16s, align [[RUST_ALIGN:2]]

    // aarch64:     [[ABI_VALUE:%.+]] = call [[ABI_TYPE]] @returns_twou16s()
    // loongarch64: [[ABI_VALUE:%.+]] = call [[ABI_TYPE]] @returns_twou16s()
    // sparc64:     [[ABI_VALUE:%.+]] = call [[ABI_TYPE]] @returns_twou16s()

    // aarch64:     store [[ABI_TYPE]] [[ABI_VALUE]], ptr [[ABI_ALLOCA]], align [[ABI_ALIGN]]
    // loongarch64: store [[ABI_TYPE]] [[ABI_VALUE]], ptr [[ABI_ALLOCA]], align [[ABI_ALIGN]]
    // sparc64:     store [[ABI_TYPE]] [[ABI_VALUE]], ptr [[ABI_ALLOCA]], align [[ABI_ALIGN]]

    // aarch64:     call void @llvm.memcpy.{{.+}}(ptr align [[RUST_ALIGN]] [[RUST_ALLOCA]], ptr align [[ABI_ALIGN]] [[ABI_ALLOCA]], i64 4, i1 false)
    // loongarch64: call void @llvm.memcpy.{{.+}}(ptr align [[RUST_ALIGN]] [[RUST_ALLOCA]], ptr align [[ABI_ALIGN]] [[ABI_ALLOCA]], i64 4, i1 false)
    // sparc64:     call void @llvm.memcpy.{{.+}}(ptr align [[RUST_ALIGN]] [[RUST_ALLOCA]], ptr align [[ABI_ALIGN]] [[ABI_ALLOCA]], i64 4, i1 false)
    returns_twou16s()
}

// CHECK-LABEL: @call_fiveu16s
#[no_mangle]
pub unsafe fn call_fiveu16s() {
    // CHECK: [[ABI_ALLOCA:%.+]] = alloca [[ABI_TYPE:\[2 x i64\]]], align [[ABI_ALIGN:8]]

    // CHECK: [[RUST_ALLOCA:%.+]] = alloca %FiveU16s, align 2

    // CHECK: call void @llvm.memcpy.{{.+}}(ptr align [[ABI_ALIGN]] [[ABI_ALLOCA]], ptr align [[RUST_ALIGN]] [[RUST_ALLOCA]], i64 10, i1 false)
    // CHECK: [[ABI_VALUE:%.+]] = load [[ABI_TYPE]], ptr [[ABI_ALLOCA]], align [[ABI_ALIGN]]
    // CHECK: call void @receives_fiveu16s([[ABI_TYPE]] [[ABI_VALUE]])
    let x = FiveU16s { a: 1, b: 2, c: 3, d: 4, e: 5 };
    receives_fiveu16s(x);
}

// CHECK-LABEL: @return_fiveu16s
// CHECK-SAME: (ptr {{.+}} sret([10 x i8]) align [[RUST_ALIGN:2]] dereferenceable(10) [[RET_PTR:%.+]])
#[no_mangle]
pub unsafe fn return_fiveu16s() -> FiveU16s {
    // powerpc returns this struct via sret pointer, it doesn't use the cast ABI.

    // powerpc64: call void @returns_fiveu16s(ptr {{.+}} [[RET_PTR]])


    // The other targets copy the cast ABI type to the sret pointer.

    // aarch64:     [[ABI_ALLOCA:%.+]] = alloca [[ABI_TYPE:\[2 x i64\]]], align [[ABI_ALIGN:8]]
    // loongarch64: [[ABI_ALLOCA:%.+]] = alloca [[ABI_TYPE:\[2 x i64\]]], align [[ABI_ALIGN:8]]
    // sparc64:     [[ABI_ALLOCA:%.+]] = alloca [[ABI_TYPE:\[2 x i64\]]], align [[ABI_ALIGN:8]]

    // aarch64:     [[ABI_VALUE:%.+]] = call [[ABI_TYPE]] @returns_fiveu16s()
    // loongarch64: [[ABI_VALUE:%.+]] = call [[ABI_TYPE]] @returns_fiveu16s()
    // sparc64:     [[ABI_VALUE:%.+]] = call [[ABI_TYPE]] @returns_fiveu16s()

    // aarch64:     store [[ABI_TYPE]] [[ABI_VALUE]], ptr [[ABI_ALLOCA]], align [[ABI_ALIGN]]
    // loongarch64: store [[ABI_TYPE]] [[ABI_VALUE]], ptr [[ABI_ALLOCA]], align [[ABI_ALIGN]]
    // sparc64:     store [[ABI_TYPE]] [[ABI_VALUE]], ptr [[ABI_ALLOCA]], align [[ABI_ALIGN]]

    // aarch64:     call void @llvm.memcpy.{{.+}}(ptr align [[RUST_ALIGN]] [[RET_PTR]], ptr align [[ABI_ALIGN]] [[ABI_ALLOCA]], i64 10, i1 false)
    // loongarch64: call void @llvm.memcpy.{{.+}}(ptr align [[RUST_ALIGN]] [[RET_PTR]], ptr align [[ABI_ALIGN]] [[ABI_ALLOCA]], i64 10, i1 false)
    // sparc64:     call void @llvm.memcpy.{{.+}}(ptr align [[RUST_ALIGN]] [[RET_PTR]], ptr align [[ABI_ALIGN]] [[ABI_ALLOCA]], i64 10, i1 false)
    returns_fiveu16s()
}

// CHECK-LABEL: @call_doubledouble
#[no_mangle]
pub unsafe fn call_doubledouble() {
    // aarch64:     [[ABI_ALLOCA:%.+]] = alloca [[ABI_TYPE:\[2 x double\]]], align [[ABI_ALIGN:8]]
    // loongarch64: [[ABI_ALLOCA:%.+]] = alloca [[ABI_TYPE:{ double, double }]], align [[ABI_ALIGN:8]]
    // powerpc64:   [[ABI_ALLOCA:%.+]] = alloca [[ABI_TYPE:\[2 x i64\]]], align [[ABI_ALIGN:8]]
    // sparc64:     [[ABI_ALLOCA:%.+]] = alloca [[ABI_TYPE:{ double, double }]], align [[ABI_ALIGN:8]]

    // CHECK: [[RUST_ALLOCA:%.+]] = alloca %DoubleDouble, align [[RUST_ALIGN:8]]

    // CHECK: call void @llvm.memcpy.{{.+}}(ptr align [[ABI_ALIGN]] [[ABI_ALLOCA]], ptr align [[RUST_ALIGN]] [[RUST_ALLOCA]], i64 16, i1 false)
    // CHECK: [[ABI_VALUE:%.+]] = load [[ABI_TYPE]], ptr [[ABI_ALLOCA]], align [[ABI_ALIGN]]
    // CHECK: call void @receives_doubledouble([[ABI_TYPE]] [[ABI_VALUE]])
    let x = DoubleDouble { f: 1., g: 2. };
    receives_doubledouble(x);
}

// CHECK-LABEL: @return_doubledouble
#[no_mangle]
pub unsafe fn return_doubledouble() -> DoubleDouble {
    // powerpc returns this struct via sret pointer, it doesn't use the cast ABI.

    // powerpc64: [[RETVAL:%.+]] = alloca %DoubleDouble, align 8
    // powerpc64: call void @returns_doubledouble(ptr {{.+}} [[RETVAL]])


    // The other targets copy the cast ABI type to an alloca.

    // aarch64:     [[ABI_ALLOCA:%.+]] = alloca [[ABI_TYPE:\[2 x double\]]], align [[ABI_ALIGN:8]]
    // loongarch64: [[ABI_ALLOCA:%.+]] = alloca [[ABI_TYPE:{ double, double }]], align [[ABI_ALIGN:8]]
    // sparc64:     [[ABI_ALLOCA:%.+]] = alloca [[ABI_TYPE:{ double, double }]], align [[ABI_ALIGN:8]]

    // aarch64:     [[RUST_ALLOCA:%.+]] = alloca %DoubleDouble, align [[RUST_ALIGN:8]]
    // loongarch64: [[RUST_ALLOCA:%.+]] = alloca %DoubleDouble, align [[RUST_ALIGN:8]]
    // sparc64:     [[RUST_ALLOCA:%.+]] = alloca %DoubleDouble, align [[RUST_ALIGN:8]]

    // aarch64:     [[ABI_VALUE:%.+]] = call [[ABI_TYPE]] @returns_doubledouble()
    // loongarch64: [[ABI_VALUE:%.+]] = call [[ABI_TYPE]] @returns_doubledouble()
    // sparc64:     [[ABI_VALUE:%.+]] = call [[ABI_TYPE]] @returns_doubledouble()

    // aarch64:     store [[ABI_TYPE]] [[ABI_VALUE]], ptr [[ABI_ALLOCA]], align [[ABI_ALIGN]]
    // loongarch64: store [[ABI_TYPE]] [[ABI_VALUE]], ptr [[ABI_ALLOCA]], align [[ABI_ALIGN]]
    // sparc64:     store [[ABI_TYPE]] [[ABI_VALUE]], ptr [[ABI_ALLOCA]], align [[ABI_ALIGN]]

    // aarch64:     call void @llvm.memcpy.{{.+}}(ptr align [[RUST_ALIGN]] [[RUST_ALLOCA]], ptr align [[ABI_ALIGN]] [[ABI_ALLOCA]], i64 16, i1 false)
    // loongarch64: call void @llvm.memcpy.{{.+}}(ptr align [[RUST_ALIGN]] [[RUST_ALLOCA]], ptr align [[ABI_ALIGN]] [[ABI_ALLOCA]], i64 16, i1 false)
    // sparc64:     call void @llvm.memcpy.{{.+}}(ptr align [[RUST_ALIGN]] [[RUST_ALLOCA]], ptr align [[ABI_ALIGN]] [[ABI_ALLOCA]], i64 16, i1 false)
    returns_doubledouble()
}

// CHECK-LABEL: @call_doublefloat
#[no_mangle]
pub unsafe fn call_doublefloat() {
    // aarch64:     [[ABI_ALLOCA:%.+]] = alloca [[ABI_TYPE:\[2 x i64\]]], align [[ABI_ALIGN:8]]
    // loongarch64: [[ABI_ALLOCA:%.+]] = alloca [[ABI_TYPE:{ double, float }]], align [[ABI_ALIGN:8]]
    // powerpc64:   [[ABI_ALLOCA:%.+]] = alloca [[ABI_TYPE:\[2 x i64\]]], align [[ABI_ALIGN:8]]
    // sparc64:     [[ABI_ALLOCA:%.+]] = alloca [[ABI_TYPE:{ double, float, i32, i64 }]], align [[ABI_ALIGN:8]]

    // CHECK: [[RUST_ALLOCA:%.+]] = alloca %DoubleFloat, align [[RUST_ALIGN:8]]

    // aarch64:     call void @llvm.memcpy.{{.+}}(ptr align [[ABI_ALIGN]] [[ABI_ALLOCA]], ptr align [[RUST_ALIGN]] [[RUST_ALLOCA]], i64 16, i1 false)
    // loongarch64: call void @llvm.memcpy.{{.+}}(ptr align [[ABI_ALIGN]] [[ABI_ALLOCA]], ptr align [[RUST_ALIGN]] [[RUST_ALLOCA]], i64 12, i1 false)
    // powerpc64:   call void @llvm.memcpy.{{.+}}(ptr align [[ABI_ALIGN]] [[ABI_ALLOCA]], ptr align [[RUST_ALIGN]] [[RUST_ALLOCA]], i64 16, i1 false)
    // sparc64:     call void @llvm.memcpy.{{.+}}(ptr align [[ABI_ALIGN]] [[ABI_ALLOCA]], ptr align [[RUST_ALIGN]] [[RUST_ALLOCA]], i64 16, i1 false)

    // CHECK: [[ABI_VALUE:%.+]] = load [[ABI_TYPE]], ptr [[ABI_ALLOCA]], align [[ABI_ALIGN]]
    // CHECK: call void @receives_doublefloat([[ABI_TYPE]] {{(inreg )?}}[[ABI_VALUE]])
    let x = DoubleFloat { f: 1., g: 2. };
    receives_doublefloat(x);
}

// CHECK-LABEL: @return_doublefloat
#[no_mangle]
pub unsafe fn return_doublefloat() -> DoubleFloat {
    // powerpc returns this struct via sret pointer, it doesn't use the cast ABI.

    // powerpc64: [[RETVAL:%.+]] = alloca %DoubleFloat, align 8
    // powerpc64: call void @returns_doublefloat(ptr {{.+}} [[RETVAL]])


    // The other targets copy the cast ABI type to an alloca.

    // aarch64:     [[ABI_ALLOCA:%.+]] = alloca [[ABI_TYPE:\[2 x i64\]]], align [[ABI_ALIGN:8]]
    // loongarch64: [[ABI_ALLOCA:%.+]] = alloca [[ABI_TYPE:{ double, float }]], align [[ABI_ALIGN:8]]
    // sparc64:     [[ABI_ALLOCA:%.+]] = alloca [[ABI_TYPE:{ double, float, i32, i64 }]], align [[ABI_ALIGN:8]]

    // aarch64:     [[RUST_ALLOCA:%.+]] = alloca %DoubleFloat, align [[RUST_ALIGN:8]]
    // loongarch64: [[RUST_ALLOCA:%.+]] = alloca %DoubleFloat, align [[RUST_ALIGN:8]]
    // sparc64:     [[RUST_ALLOCA:%.+]] = alloca %DoubleFloat, align [[RUST_ALIGN:8]]

    // aarch64:     [[ABI_VALUE:%.+]] = call [[ABI_TYPE]] @returns_doublefloat()
    // loongarch64: [[ABI_VALUE:%.+]] = call [[ABI_TYPE]] @returns_doublefloat()
    // sparc64:     [[ABI_VALUE:%.+]] = call inreg [[ABI_TYPE]] @returns_doublefloat()

    // aarch64:     store [[ABI_TYPE]] [[ABI_VALUE]], ptr [[ABI_ALLOCA]], align [[ABI_ALIGN]]
    // loongarch64: store [[ABI_TYPE]] [[ABI_VALUE]], ptr [[ABI_ALLOCA]], align [[ABI_ALIGN]]
    // sparc64:     store [[ABI_TYPE]] [[ABI_VALUE]], ptr [[ABI_ALLOCA]], align [[ABI_ALIGN]]

    // aarch64:     call void @llvm.memcpy.{{.+}}(ptr align [[RUST_ALIGN]] [[RUST_ALLOCA]], ptr align [[ABI_ALIGN]] [[ABI_ALLOCA]], i64 16, i1 false)
    // loongarch64: call void @llvm.memcpy.{{.+}}(ptr align [[RUST_ALIGN]] [[RUST_ALLOCA]], ptr align [[ABI_ALIGN]] [[ABI_ALLOCA]], i64 12, i1 false)
    // sparc64:     call void @llvm.memcpy.{{.+}}(ptr align [[RUST_ALIGN]] [[RUST_ALLOCA]], ptr align [[ABI_ALIGN]] [[ABI_ALLOCA]], i64 16, i1 false)
    returns_doublefloat()
}
