// revisions: x32 x64 sparc sparc64
// compile-flags: -O -C no-prepopulate-passes
//
//[x32] only-x86
//[x64] only-x86_64
//[sparc] only-sparc
//[sparc64] only-sparc64
// ignore-windows
// See ./transparent.rs
// Some platforms pass large aggregates using immediate arrays in LLVMIR
// Other platforms pass large aggregates using struct pointer in LLVMIR
// This covers the "struct pointer" case.


#![feature(transparent_unions)]

#![crate_type="lib"]


#[derive(Clone, Copy)]
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

// CHECK: define{{.*}}void @test_BigS(ptr [[BIGS_RET_ATTRS1:.*]] sret(%BigS) [[BIGS_RET_ATTRS2:.*]], ptr [[BIGS_ARG_ATTRS1:.*]] byval(%BigS) [[BIGS_ARG_ATTRS2:.*]])
#[no_mangle]
pub extern "C" fn test_BigS(_: BigS) -> BigS { loop {} }

// CHECK: define{{.*}}void @test_TsBigS(ptr [[BIGS_RET_ATTRS1]] sret(%TsBigS) [[BIGS_RET_ATTRS2]], ptr [[BIGS_ARG_ATTRS1]] byval(%TsBigS) [[BIGS_ARG_ATTRS2:.*]])
#[no_mangle]
pub extern "C" fn test_TsBigS(_: TsBigS) -> TsBigS { loop {} }

// CHECK: define{{.*}}void @test_TuBigS(ptr [[BIGS_RET_ATTRS1]] sret(%TuBigS) [[BIGS_RET_ATTRS2]], ptr [[BIGS_ARG_ATTRS1]] byval(%TuBigS) [[BIGS_ARG_ATTRS2:.*]])
#[no_mangle]
pub extern "C" fn test_TuBigS(_: TuBigS) -> TuBigS { loop {} }

// CHECK: define{{.*}}void @test_TeBigS(ptr [[BIGS_RET_ATTRS1]] sret(%"TeBigS::Variant") [[BIGS_RET_ATTRS2]], ptr [[BIGS_ARG_ATTRS1]] byval(%"TeBigS::Variant") [[BIGS_ARG_ATTRS2]])
#[no_mangle]
pub extern "C" fn test_TeBigS(_: TeBigS) -> TeBigS { loop {} }


#[derive(Clone, Copy)]
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

// CHECK: define{{.*}}void @test_BigU(ptr [[BIGU_RET_ATTRS1:.*]] sret(%BigU) [[BIGU_RET_ATTRS2:.*]], ptr [[BIGU_ARG_ATTRS1:.*]] byval(%BigU) [[BIGU_ARG_ATTRS2:.*]])
#[no_mangle]
pub extern "C" fn test_BigU(_: BigU) -> BigU { loop {} }

// CHECK: define{{.*}}void @test_TsBigU(ptr [[BIGU_RET_ATTRS1:.*]] sret(%TsBigU) [[BIGU_RET_ATTRS2:.*]], ptr [[BIGU_ARG_ATTRS1]] byval(%TsBigU) [[BIGU_ARG_ATTRS2]])
#[no_mangle]
pub extern "C" fn test_TsBigU(_: TsBigU) -> TsBigU { loop {} }

// CHECK: define{{.*}}void @test_TuBigU(ptr [[BIGU_RET_ATTRS1]] sret(%TuBigU) [[BIGU_RET_ATTRS2:.*]], ptr [[BIGU_ARG_ATTRS1]] byval(%TuBigU) [[BIGU_ARG_ATTRS2]])
#[no_mangle]
pub extern "C" fn test_TuBigU(_: TuBigU) -> TuBigU { loop {} }

// CHECK: define{{.*}}void @test_TeBigU(ptr [[BIGU_RET_ATTRS1]] sret(%"TeBigU::Variant") [[BIGU_RET_ATTRS2:.*]], ptr [[BIGU_ARG_ATTRS1]] byval(%"TeBigU::Variant") [[BIGU_ARG_ATTRS2]])
#[no_mangle]
pub extern "C" fn test_TeBigU(_: TeBigU) -> TeBigU { loop {} }
