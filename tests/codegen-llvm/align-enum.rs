//@ compile-flags: -C no-prepopulate-passes -Z mir-opt-level=0
//

#![crate_type = "lib"]

#[repr(align(64))]
pub enum Align64 {
    A(u32),
    B(u32),
}

pub struct Nested64 {
    a: u8,
    b: Align64,
    c: u16,
}

// CHECK-LABEL: @align64
#[no_mangle]
pub fn align64(a: u32) -> Align64 {
    // CHECK: %a64 = alloca [64 x i8], align 64
    // CHECK: call void @llvm.memcpy.{{.*}}(ptr align 64 %{{.*}}, ptr align 64 %{{.*}}, i{{[0-9]+}} 64, i1 false)
    let a64 = Align64::A(a);
    a64
}

// CHECK-LABEL: @nested64
#[no_mangle]
pub fn nested64(a: u8, b: u32, c: u16) -> Nested64 {
    // CHECK: %n64 = alloca [128 x i8], align 64
    let n64 = Nested64 { a, b: Align64::B(b), c };
    n64
}
