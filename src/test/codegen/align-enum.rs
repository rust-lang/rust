// compile-flags: -C no-prepopulate-passes
// ignore-tidy-linelength
// min-llvm-version 7.0

#![crate_type = "lib"]

#[repr(align(64))]
pub enum Align64 {
    A(u32),
    B(u32),
}
// CHECK: %Align64 = type { [0 x i32], i32, [15 x i32] }

pub struct Nested64 {
    a: u8,
    b: Align64,
    c: u16,
}

// CHECK-LABEL: @align64
#[no_mangle]
pub fn align64(a: u32) -> Align64 {
// CHECK: %a64 = alloca %Align64, align 64
// CHECK: call void @llvm.memcpy.{{.*}}(i8* align 64 %{{.*}}, i8* align 64 %{{.*}}, i{{[0-9]+}} 64, i1 false)
    let a64 = Align64::A(a);
    a64
}

// CHECK-LABEL: @nested64
#[no_mangle]
pub fn nested64(a: u8, b: u32, c: u16) -> Nested64 {
// CHECK: %n64 = alloca %Nested64, align 64
    let n64 = Nested64 { a, b: Align64::B(b), c };
    n64
}
