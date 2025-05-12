// Make sure that no 0-sized padding is inserted in structs and that
// structs are represented as expected by Neon intrinsics in LLVM.
// See #87254.

#![crate_type = "lib"]
#![feature(repr_simd, abi_unadjusted)]

#[derive(Copy, Clone)]
#[repr(simd)]
pub struct int16x4_t(pub [i16; 4]);

#[derive(Copy, Clone)]
pub struct int16x4x2_t(pub int16x4_t, pub int16x4_t);

// CHECK: %int16x4x2_t = type { <4 x i16>, <4 x i16> }
#[no_mangle]
extern "unadjusted" fn takes_int16x4x2_t(t: int16x4x2_t) -> int16x4x2_t {
    t
}
