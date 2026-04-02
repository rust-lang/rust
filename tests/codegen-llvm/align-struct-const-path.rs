//@ compile-flags: -C no-prepopulate-passes -Z mir-opt-level=0
// 32bit MSVC does not align things properly so we suppress high alignment annotations (#112480)
//@ ignore-i686-pc-windows-msvc
//@ ignore-i686-pc-windows-gnu

#![crate_type = "lib"]
#![feature(const_attr_paths)]

mod alignments {
    pub const STRUCT: usize = 64;
}

#[repr(align(alignments::STRUCT))]
pub struct Align64(i32);

// CHECK-LABEL: @align64_const
#[no_mangle]
pub fn align64_const(i: i32) -> Align64 {
    // CHECK: %a64 = alloca [64 x i8], align 64
    let a64 = Align64(i);
    a64
}
