//@ only-aarch64
#![feature(rustc_attrs)]
#![crate_type = "lib"]

// Tests that the computed layout size of scalable vectors is equal to
// `element size * element count * number of vectors`. Scalable vectors are of course scalable and
// so do not have a fixed size, but using this size with things like `llvm.memcpy` produces the
// correct and expected results.

#[rustc_dump_layout(size)]
#[rustc_scalable_vector(4)]
struct ScalableFloat(f32); //~ ERROR: size: Size(16 bytes)

#[rustc_dump_layout(size)]
#[rustc_scalable_vector(8)]
struct ScalableU8WithFewerCount(u8); //~ ERROR: size: Size(8 bytes)

#[rustc_dump_layout(size)]
#[rustc_scalable_vector(16)]
struct ScalableU8(u8); //~ ERROR: size: Size(16 bytes)

#[rustc_dump_layout(size)]
#[rustc_scalable_vector(16)]
struct ScalableBool(bool); //~ ERROR: size: Size(16 bytes)

#[rustc_dump_layout(size)]
#[rustc_scalable_vector]
struct ScalableTuple(ScalableU8, ScalableU8, ScalableU8); //~ ERROR: size: Size(48 bytes)
