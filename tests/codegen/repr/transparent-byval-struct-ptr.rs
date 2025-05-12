//@ add-core-stubs
//@ revisions: i686-linux i686-freebsd x64-linux x64-apple
//@ compile-flags: -Copt-level=3 -C no-prepopulate-passes

//@[i686-linux] compile-flags: --target i686-unknown-linux-gnu
//@[i686-linux] needs-llvm-components: x86
//@[i686-freebsd] compile-flags: --target i686-unknown-freebsd
//@[i686-freebsd] needs-llvm-components: x86
//@[x64-linux] compile-flags: --target x86_64-unknown-linux-gnu
//@[x64-linux] needs-llvm-components: x86
//@[x64-apple] compile-flags: --target x86_64-apple-darwin
//@[x64-apple] needs-llvm-components: x86

// See ./transparent.rs
// Some platforms pass large aggregates using immediate arrays in LLVMIR
// Other platforms pass large aggregates using by-value struct pointer in LLVMIR
// Yet more platforms pass large aggregates using opaque pointer in LLVMIR
// This covers the "by-value struct pointer" case.

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

// CHECK: define{{.*}}void @test_BigS(ptr [[BIGS_RET_ATTRS1:.*]] sret([64 x i8]) [[BIGS_RET_ATTRS2:.*]], ptr [[BIGS_ARG_ATTRS1:.*]] byval([64 x i8]) [[BIGS_ARG_ATTRS2:.*]])
#[no_mangle]
pub extern "C" fn test_BigS(_: BigS) -> BigS {
    loop {}
}

// CHECK: define{{.*}}void @test_TsBigS(ptr [[BIGS_RET_ATTRS1]] sret([64 x i8]) [[BIGS_RET_ATTRS2]], ptr [[BIGS_ARG_ATTRS1]] byval([64 x i8]) [[BIGS_ARG_ATTRS2:.*]])
#[no_mangle]
pub extern "C" fn test_TsBigS(_: TsBigS) -> TsBigS {
    loop {}
}

// CHECK: define{{.*}}void @test_TuBigS(ptr [[BIGS_RET_ATTRS1]] sret([64 x i8]) [[BIGS_RET_ATTRS2]], ptr [[BIGS_ARG_ATTRS1]] byval([64 x i8]) [[BIGS_ARG_ATTRS2:.*]])
#[no_mangle]
pub extern "C" fn test_TuBigS(_: TuBigS) -> TuBigS {
    loop {}
}

// CHECK: define{{.*}}void @test_TeBigS(ptr [[BIGS_RET_ATTRS1]] sret([64 x i8]) [[BIGS_RET_ATTRS2]], ptr [[BIGS_ARG_ATTRS1]] byval([64 x i8]) [[BIGS_ARG_ATTRS2]])
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

// CHECK: define{{.*}}void @test_BigU(ptr [[BIGU_RET_ATTRS1:.*]] sret([64 x i8]) [[BIGU_RET_ATTRS2:.*]], ptr [[BIGU_ARG_ATTRS1:.*]] byval([64 x i8]) [[BIGU_ARG_ATTRS2:.*]])
#[no_mangle]
pub extern "C" fn test_BigU(_: BigU) -> BigU {
    loop {}
}

// CHECK: define{{.*}}void @test_TsBigU(ptr [[BIGU_RET_ATTRS1:.*]] sret([64 x i8]) [[BIGU_RET_ATTRS2:.*]], ptr [[BIGU_ARG_ATTRS1]] byval([64 x i8]) [[BIGU_ARG_ATTRS2]])
#[no_mangle]
pub extern "C" fn test_TsBigU(_: TsBigU) -> TsBigU {
    loop {}
}

// CHECK: define{{.*}}void @test_TuBigU(ptr [[BIGU_RET_ATTRS1]] sret([64 x i8]) [[BIGU_RET_ATTRS2:.*]], ptr [[BIGU_ARG_ATTRS1]] byval([64 x i8]) [[BIGU_ARG_ATTRS2]])
#[no_mangle]
pub extern "C" fn test_TuBigU(_: TuBigU) -> TuBigU {
    loop {}
}

// CHECK: define{{.*}}void @test_TeBigU(ptr [[BIGU_RET_ATTRS1]] sret([64 x i8]) [[BIGU_RET_ATTRS2:.*]], ptr [[BIGU_ARG_ATTRS1]] byval([64 x i8]) [[BIGU_ARG_ATTRS2]])
#[no_mangle]
pub extern "C" fn test_TeBigU(_: TeBigU) -> TeBigU {
    loop {}
}
