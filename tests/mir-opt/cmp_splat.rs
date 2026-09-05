//@ skip-filecheck

#![feature(splat)]
#![feature(tuple_trait)]

// Comparing the MIR of a splat-compatible function with it's direct counterpart.
// It's expected that there will be additional MIR generated due to the layer of
// indirection added by a trait (MinArgs). This test tracks that overhead.

use std::marker::Tuple;

// EMIT_MIR cmp_splat.min_direct_u8.built.after.mir
pub fn min_direct_u8(x: u8, y: u8) -> u8 {
    min_direct(x, y)
}

// EMIT_MIR cmp_splat.min_splatted_u8.built.after.mir
pub fn min_splatted_u8(x: u8, y: u8) -> u8 {
    min_splatted(x, y)
}

// EMIT_MIR cmp_splat.min_direct.built.after.mir
#[inline]
pub fn min_direct<T: Ord>(v1: T, v2: T) -> T {
    v1.min(v2)
}

// EMIT_MIR cmp_splat.min_splatted.built.after.mir
#[inline]
pub fn min_splatted<T: Ord>(v1: T, v2: T, #[rustc_splat] args: impl MinArgs<T>) -> T {
    MinArgs::min_splatted_inner(v1, v2, args)
}

trait MinArgs<T>: Tuple {
    fn min_splatted_inner(v1: T, v2: T, args: Self) -> T;
}

// EMIT_MIR cmp_splat.{impl#0}-min_splatted_inner.built.after.mir
impl<T: Ord> MinArgs<T> for () {
    #[inline(always)]
    fn min_splatted_inner(v1: T, v2: T, (): Self) -> T {
        v1.min(v2)
    }
}
