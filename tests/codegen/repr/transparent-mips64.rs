//@ revisions: mips64 mips64el
//@ compile-flags: -O -C no-prepopulate-passes

//@[mips64] compile-flags: --target mips64-unknown-linux-gnuabi64
//@[mips64] needs-llvm-components: mips
//@[mips64el] compile-flags: --target mips64el-unknown-linux-gnuabi64
//@[mips64el] needs-llvm-components: mips

// See ./transparent.rs

#![feature(no_core, lang_items, transparent_unions)]
#![crate_type = "lib"]
#![no_std]
#![no_core]

#[lang="sized"] trait Sized { }
#[lang="freeze"] trait Freeze { }
#[lang="copy"] trait Copy { }

impl Copy for [u32; 16] {}
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

// CHECK: define void @test_BigS(ptr [[BIGS_RET_ATTRS1:.*]] sret([64 x i8]) [[BIGS_RET_ATTRS2:.*]], [8 x i64]
#[no_mangle]
pub extern fn test_BigS(_: BigS) -> BigS { loop {} }

// CHECK: define void @test_TsBigS(ptr [[BIGS_RET_ATTRS1]] sret([64 x i8]) [[BIGS_RET_ATTRS2]], [8 x i64]
#[no_mangle]
pub extern fn test_TsBigS(_: TsBigS) -> TsBigS { loop {} }

// CHECK: define void @test_TuBigS(ptr [[BIGS_RET_ATTRS1]] sret([64 x i8]) [[BIGS_RET_ATTRS2]], [8 x i64]
#[no_mangle]
pub extern fn test_TuBigS(_: TuBigS) -> TuBigS { loop {} }

// CHECK: define void @test_TeBigS(ptr [[BIGS_RET_ATTRS1]] sret([64 x i8]) [[BIGS_RET_ATTRS2]], [8 x i64]
#[no_mangle]
pub extern fn test_TeBigS(_: TeBigS) -> TeBigS { loop {} }


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

// CHECK: define void @test_BigU(ptr [[BIGU_RET_ATTRS1:.*]] sret([64 x i8]) [[BIGU_RET_ATTRS2:.*]], [8 x i64]
#[no_mangle]
pub extern fn test_BigU(_: BigU) -> BigU { loop {} }

// CHECK: define void @test_TsBigU(ptr [[BIGU_RET_ATTRS1]] sret([64 x i8]) [[BIGU_RET_ATTRS2]], [8 x i64]
#[no_mangle]
pub extern fn test_TsBigU(_: TsBigU) -> TsBigU { loop {} }

// CHECK: define void @test_TuBigU(ptr [[BIGU_RET_ATTRS1]] sret([64 x i8]) [[BIGU_RET_ATTRS2]], [8 x i64]
#[no_mangle]
pub extern fn test_TuBigU(_: TuBigU) -> TuBigU { loop {} }

// CHECK: define void @test_TeBigU(ptr [[BIGU_RET_ATTRS1]] sret([64 x i8]) [[BIGU_RET_ATTRS2]], [8 x i64]
#[no_mangle]
pub extern fn test_TeBigU(_: TeBigU) -> TeBigU { loop {} }
