//@ add-core-stubs
//@ revisions: arm-linux arm-android armv7-linux armv7-android mips thumb sparc
//@ compile-flags: -Copt-level=3 -C no-prepopulate-passes

//@[arm-linux] compile-flags: --target arm-unknown-linux-gnueabi
//@[arm-linux] needs-llvm-components: arm
//@[arm-android] compile-flags: --target arm-linux-androideabi
//@[arm-android] needs-llvm-components: arm
//@[armv7-linux] compile-flags: --target armv7-unknown-linux-gnueabi
//@[armv7-linux] needs-llvm-components: arm
//@[armv7-android] compile-flags: --target armv7-linux-androideabi
//@[armv7-android] needs-llvm-components: arm
//@[mips] compile-flags: --target mips-unknown-linux-gnu
//@[mips] needs-llvm-components: mips
//@[thumb] compile-flags: --target thumbv7neon-linux-androideabi
//@[thumb] needs-llvm-components: arm
//@[sparc] compile-flags: --target sparc-unknown-linux-gnu
//@[sparc] needs-llvm-components: sparc

// See ./transparent.rs
// Some platforms pass large aggregates using immediate arrays in LLVMIR
// Other platforms pass large aggregates using by-value struct pointer in LLVMIR
// Yet more platforms pass large aggregates using opaque pointer in LLVMIR
// This covers the "immediate array" case.

#![feature(no_core, lang_items, transparent_unions)]
#![crate_type = "lib"]
#![no_std]
#![no_core]

extern crate minicore;
use minicore::*;
impl Copy for BigS {}
impl Copy for BigU {}

#[repr(C)]
pub struct BigS([u32; 16]);

#[repr(transparent)]
pub struct TsBigS(BigS);

#[repr(transparent)]
pub union TuBigS {
    field: BigS,
}

#[repr(transparent)]
pub enum TeBigS {
    Variant(BigS),
}

// CHECK: define void @test_BigS(ptr [[BIGS_RET_ATTRS1:.*]] sret([64 x i8]) [[BIGS_RET_ATTRS2:.*]], [16 x i32]
#[no_mangle]
pub extern "C" fn test_BigS(_: BigS) -> BigS {
    loop {}
}

// CHECK: define void @test_TsBigS(ptr [[BIGS_RET_ATTRS1]] sret([64 x i8]) [[BIGS_RET_ATTRS2]], [16 x i32]
#[no_mangle]
pub extern "C" fn test_TsBigS(_: TsBigS) -> TsBigS {
    loop {}
}

// CHECK: define void @test_TuBigS(ptr [[BIGS_RET_ATTRS1]] sret([64 x i8]) [[BIGS_RET_ATTRS2]], [16 x i32]
#[no_mangle]
pub extern "C" fn test_TuBigS(_: TuBigS) -> TuBigS {
    loop {}
}

// CHECK: define void @test_TeBigS(ptr [[BIGS_RET_ATTRS1]] sret([64 x i8]) [[BIGS_RET_ATTRS2]], [16 x i32]
#[no_mangle]
pub extern "C" fn test_TeBigS(_: TeBigS) -> TeBigS {
    loop {}
}

#[repr(C)]
pub union BigU {
    foo: [u32; 16],
}

#[repr(transparent)]
pub struct TsBigU(BigU);

#[repr(transparent)]
pub union TuBigU {
    field: BigU,
}

#[repr(transparent)]
pub enum TeBigU {
    Variant(BigU),
}

// CHECK: define void @test_BigU(ptr [[BIGU_RET_ATTRS1:.*]] sret([64 x i8]) [[BIGU_RET_ATTRS2:.*]], [16 x i32]
#[no_mangle]
pub extern "C" fn test_BigU(_: BigU) -> BigU {
    loop {}
}

// CHECK: define void @test_TsBigU(ptr [[BIGU_RET_ATTRS1]] sret([64 x i8]) [[BIGU_RET_ATTRS2]], [16 x i32]
#[no_mangle]
pub extern "C" fn test_TsBigU(_: TsBigU) -> TsBigU {
    loop {}
}

// CHECK: define void @test_TuBigU(ptr [[BIGU_RET_ATTRS1]] sret([64 x i8]) [[BIGU_RET_ATTRS2]], [16 x i32]
#[no_mangle]
pub extern "C" fn test_TuBigU(_: TuBigU) -> TuBigU {
    loop {}
}

// CHECK: define void @test_TeBigU(ptr [[BIGU_RET_ATTRS1]] sret([64 x i8]) [[BIGU_RET_ATTRS2]], [16 x i32]
#[no_mangle]
pub extern "C" fn test_TeBigU(_: TeBigU) -> TeBigU {
    loop {}
}
