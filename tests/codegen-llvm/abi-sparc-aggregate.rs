// Checks that 32-bit SPARC passes aggregate arguments by reference, matching GCC and Clang
// (`SparcV8ABIInfo` falls back to Clang's `DefaultABIInfo::classifyArgumentType`, which makes
// aggregates `byval`). Regression test for rustc passing them as a run of `i32` registers.

//@ add-minicore
//@ compile-flags: -Copt-level=3 -C no-prepopulate-passes --target=sparc-unknown-linux-gnu
//@ needs-llvm-components: sparc

#![feature(no_core, lang_items)]
#![crate_type = "lib"]
#![no_core]

extern crate minicore;
use minicore::*;

#[repr(C)]
pub struct S4 {
    a: u32,
}

#[repr(C)]
pub struct S16 {
    a: u32,
    b: u64,
}

#[repr(C)]
pub struct S240 {
    a: [u32; 60],
}

// CHECK-LABEL: define void @take_s4(ptr {{.*}}byval([4 x i8]) align 4 {{.*}})
#[no_mangle]
pub extern "C" fn take_s4(_: S4) {}

// CHECK-LABEL: define void @take_s16(ptr {{.*}}byval([16 x i8]) align 8 {{.*}})
#[no_mangle]
pub extern "C" fn take_s16(_: S16) {}

// CHECK-LABEL: define void @take_s240(ptr {{.*}}byval([240 x i8]) align 4 {{.*}})
#[no_mangle]
pub extern "C" fn take_s240(_: S240) {}

// An aggregate return keeps using `sret`.
// CHECK-LABEL: define void @roundtrip_s16(ptr {{.*}}sret([16 x i8]) align 8 {{.*}}, ptr {{.*}}byval([16 x i8]) align 8 {{.*}})
#[no_mangle]
pub extern "C" fn roundtrip_s16(s: S16) -> S16 {
    s
}

extern "C" {
    fn extern_take_s16(s: S16);
}

// CHECK-LABEL: define void @call_s16(
// CHECK: call void @extern_take_s16(ptr {{.*}}byval([16 x i8]) align 8 {{.*}})
#[no_mangle]
pub extern "C" fn call_s16(s: S16) {
    unsafe { extern_take_s16(s) }
}

// Scalars keep being passed directly, and small integers are sign/zero extended to 32 bits.
// CHECK-LABEL: define void @take_scalars(i8 noundef signext %_a, i16 noundef zeroext %_b, i32 noundef %_c)
#[no_mangle]
pub extern "C" fn take_scalars(_a: i8, _b: u16, _c: u32) {}
