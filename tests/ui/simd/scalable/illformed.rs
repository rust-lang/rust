//@ compile-flags: --crate-type=lib
#![feature(repr_scalable, repr_simd)]

#[repr(simd, scalable(4))]
struct NoFields {} //~ ERROR: scalable vectors must have a single field

#[repr(simd, scalable(4))]
struct MultipleFields { //~ ERROR: scalable vectors cannot have multiple fields
    _ty: [f32], //~ ERROR: the size for values of type `[f32]` cannot be known at compilation time
    other: u32,
}

#[repr(simd, scalable(4))]
struct WrongFieldType { //~ ERROR: the field of a scalable vector type must be a slice
    _ty: String,
}

#[repr(simd, scalable(4))]
struct WrongElementTy { //~ ERROR: element type of a scalable vector must be a primitive scalar
    _ty: [String],
}
