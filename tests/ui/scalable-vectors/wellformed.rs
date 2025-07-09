//@ check-pass
//@ compile-flags: --crate-type=lib
#![feature(rustc_attrs)]

#[rustc_scalable_vector(16)]
struct ScalableU8(u8);

#[rustc_scalable_vector(8)]
struct ScalableU16(u16);

#[rustc_scalable_vector(4)]
struct ScalableU32(u32);

#[rustc_scalable_vector(2)]
struct ScalableU64(u64);

#[rustc_scalable_vector(1)]
struct ScalableU128(u128);

#[rustc_scalable_vector(16)]
struct ScalableI8(i8);

#[rustc_scalable_vector(8)]
struct ScalableI16(i16);

#[rustc_scalable_vector(4)]
struct ScalableI32(i32);

#[rustc_scalable_vector(2)]
struct ScalableI64(i64);

#[rustc_scalable_vector(1)]
struct ScalableI128(i128);

#[rustc_scalable_vector(8)]
struct ScalableF16(f32);

#[rustc_scalable_vector(4)]
struct ScalableF32(f32);

#[rustc_scalable_vector(2)]
struct ScalableF64(f64);

#[rustc_scalable_vector(16)]
struct ScalableBool(bool);

#[rustc_scalable_vector]
struct ScalableTuple(ScalableU8, ScalableU8, ScalableU8);
