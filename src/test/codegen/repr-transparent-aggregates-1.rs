// compile-flags: -C no-prepopulate-passes

// ignore-arm
// ignore-mips
// ignore-mips64
// ignore-powerpc
// ignore-powerpc64
// See repr-transparent.rs

#![crate_type="lib"]


#[repr(C)]
pub struct Big([u32; 16]);

#[repr(transparent)]
pub struct BigW(Big);

// CHECK: define void @test_Big(%Big* [[BIG_RET_ATTRS:.*]], %Big* [[BIG_ARG_ATTRS:.*]])
#[no_mangle]
pub extern fn test_Big(_: Big) -> Big { loop {} }

// CHECK: define void @test_BigW(%BigW* [[BIG_RET_ATTRS]], %BigW* [[BIG_ARG_ATTRS]])
#[no_mangle]
pub extern fn test_BigW(_: BigW) -> BigW { loop {} }


#[repr(C)]
pub union BigU {
    foo: [u32; 16],
}

#[repr(transparent)]
pub struct BigUw(BigU);

// CHECK: define void @test_BigU(%BigU* [[BIGU_RET_ATTRS:.*]], %BigU* [[BIGU_ARG_ATTRS:.*]])
#[no_mangle]
pub extern fn test_BigU(_: BigU) -> BigU { loop {} }

// CHECK: define void @test_BigUw(%BigUw* [[BIGU_RET_ATTRS]], %BigUw* [[BIGU_ARG_ATTRS]])
#[no_mangle]
pub extern fn test_BigUw(_: BigUw) -> BigUw { loop {} }
