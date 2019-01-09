// compile-flags: -C no-prepopulate-passes
// ignore-tidy-linelength
// min-llvm-version 7.0

#![crate_type = "lib"]

#[repr(align(64))]
pub struct Align64(i32);
// CHECK: %Align64 = type { [0 x i32], i32, [15 x i32] }

pub struct Nested64 {
    a: Align64,
    b: i32,
    c: i32,
    d: i8,
}
// CHECK: %Nested64 = type { [0 x i64], %Align64, [0 x i32], i32, [0 x i32], i32, [0 x i8], i8, [55 x i8] }

pub enum Enum4 {
    A(i32),
    B(i32),
}
// CHECK: %"Enum4::A" = type { [1 x i32], i32, [0 x i32] }

pub enum Enum64 {
    A(Align64),
    B(i32),
}
// CHECK: %Enum64 = type { [0 x i32], i32, [31 x i32] }
// CHECK: %"Enum64::A" = type { [8 x i64], %Align64, [0 x i64] }

// CHECK-LABEL: @align64
#[no_mangle]
pub fn align64(i : i32) -> Align64 {
// CHECK: %a64 = alloca %Align64, align 64
// CHECK: call void @llvm.memcpy.{{.*}}(i8* align 64 %{{.*}}, i8* align 64 %{{.*}}, i{{[0-9]+}} 64, i1 false)
    let a64 = Align64(i);
    a64
}

// For issue 54028: make sure that we are specifying the correct alignment for fields of aligned
// structs
// CHECK-LABEL: @align64_load
#[no_mangle]
pub fn align64_load(a: Align64) -> i32 {
// CHECK: [[FIELD:%.*]] = bitcast %Align64* %{{.*}} to i32*
// CHECK: {{%.*}} = load i32, i32* [[FIELD]], align 64
    a.0
}

// CHECK-LABEL: @nested64
#[no_mangle]
pub fn nested64(a: Align64, b: i32, c: i32, d: i8) -> Nested64 {
// CHECK: %n64 = alloca %Nested64, align 64
    let n64 = Nested64 { a, b, c, d };
    n64
}

// CHECK-LABEL: @enum4
#[no_mangle]
pub fn enum4(a: i32) -> Enum4 {
// CHECK: %e4 = alloca { i32, i32 }, align 4
    let e4 = Enum4::A(a);
    e4
}

// CHECK-LABEL: @enum64
#[no_mangle]
pub fn enum64(a: Align64) -> Enum64 {
// CHECK: %e64 = alloca %Enum64, align 64
    let e64 = Enum64::A(a);
    e64
}
