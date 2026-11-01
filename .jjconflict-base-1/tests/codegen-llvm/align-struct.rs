//@ compile-flags: -C no-prepopulate-passes -Z mir-opt-level=0
// 32bit MSVC does not align things properly so we suppress high alignment annotations (#112480)
//@ ignore-i686-pc-windows-msvc
//@ ignore-i686-pc-windows-gnu

#![crate_type = "lib"]

#[repr(align(64))]
pub struct Align64(i32);

pub struct Nested64 {
    a: Align64,
    b: i32,
    c: i32,
    d: i8,
}

// This has the extra field in B to ensure it's not ScalarPair,
// and thus that the test actually emits it via memory, not `insertvalue`.
pub enum Enum4 {
    A(i32),
    B(i32, i32),
}

pub enum Enum64 {
    A(Align64),
    B(i32),
}

// CHECK-LABEL: @align64
#[no_mangle]
pub fn align64(i: i32) -> Align64 {
    // CHECK: %a64 = alloca [64 x i8], align 64
    // CHECK: call void @llvm.memcpy.{{.*}}(ptr align 64 %{{.*}}, ptr align 64 %{{.*}}, i{{[0-9]+}} 64, i1 false)
    let a64 = Align64(i);
    a64
}

// For issue 54028: make sure that we are specifying the correct alignment for fields of aligned
// structs
// CHECK-LABEL: @align64_load
#[no_mangle]
pub fn align64_load(a: Align64) -> i32 {
    // CHECK: {{%.*}} = load i32, ptr {{%.*}}, align 64
    a.0
}

// CHECK-LABEL: @nested64
#[no_mangle]
pub fn nested64(a: Align64, b: i32, c: i32, d: i8) -> Nested64 {
    // CHECK: %n64 = alloca [128 x i8], align 64
    let n64 = Nested64 { a, b, c, d };
    n64
}

// CHECK-LABEL: @enum4
#[no_mangle]
pub fn enum4(a: i32) -> Enum4 {
    // CHECK: %e4 = alloca [12 x i8], align 4
    let e4 = Enum4::A(a);
    e4
}

// CHECK-LABEL: @enum64
#[no_mangle]
pub fn enum64(a: Align64) -> Enum64 {
    // CHECK: %e64 = alloca [128 x i8], align 64
    let e64 = Enum64::A(a);
    e64
}
