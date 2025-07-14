//@ check-pass
//@ compile-flags: --crate-type=lib
#![feature(repr_scalable, repr_simd)]

#[repr(simd, scalable(4))]
struct ScalableU8 {
    _ty: [u8],
}

#[repr(simd, scalable(4))]
struct ScalableU16 {
    _ty: [u16],
}

#[repr(simd, scalable(4))]
struct ScalableU32 {
    _ty: [u32],
}

#[repr(simd, scalable(4))]
struct ScalableU64 {
    _ty: [u64],
}

#[repr(simd, scalable(4))]
struct ScalableI8 {
    _ty: [i8],
}

#[repr(simd, scalable(4))]
struct ScalableI16 {
    _ty: [i16],
}

#[repr(simd, scalable(4))]
struct ScalableI32 {
    _ty: [i32],
}

#[repr(simd, scalable(4))]
struct ScalableI64 {
    _ty: [i64],
}

#[repr(simd, scalable(4))]
struct ScalableF32 {
    _ty: [f32],
}

#[repr(simd, scalable(4))]
struct ScalableF64 {
    _ty: [f64],
}

#[repr(simd, scalable(4))]
struct ScalableBool {
    _ty: [bool],
}

#[repr(simd, scalable(4))]
struct ScalableConstPtr {
    _ty: [*const u8],
}

#[repr(simd, scalable(4))]
struct ScalableMutPtr {
    _ty: [*mut u8],
}
