// Make sure that no 0-sized padding is inserted in structs and that
// structs are represented as expected by Neon intrinsics in LLVM.
// See #87254.

#![crate_type = "lib"]
#![feature(repr_simd)]

#[derive(Copy, Clone, Debug)]
#[repr(simd)]
pub struct int16x4_t(pub i16, pub i16, pub i16, pub i16);

#[derive(Copy, Clone, Debug)]
pub struct int16x4x2_t(pub int16x4_t, pub int16x4_t);
// CHECK: %int16x4x2_t = type { <4 x i16>, <4 x i16> }
