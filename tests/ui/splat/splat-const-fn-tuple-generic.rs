//@ run-pass
//! Test using `#[splat]` on tuple arguments of const functions with generics.

#![allow(incomplete_features)]
#![feature(splat)]

// Generic type in first position
const fn const_generic_first<T: Copy>(#[splat] _: (T, u32)) {}

// Generic type in second position
const fn const_generic_second<T: Copy>(#[splat] _: (u32, T)) {}

// Multiple generic types
const fn const_generic_both<T: Copy, U: Copy>(#[splat] _: (T, U)) {}

// Generic with extra non-splatted arg
const fn const_generic_extra<T: Copy>(#[splat] _: (T, u32), _extra: i32) {}

fn main() {
    const_generic_first(1i8, 2u32);
    const_generic_first(true, 2u32);
    const_generic_first(1u64, 2u32);

    const_generic_second(1u32, 2i8);
    const_generic_second(1u32, true);
    const_generic_second(1u32, 2u64);

    const_generic_both(1u32, 2i8);
    const_generic_both(true, 2u64);
    const_generic_both(1i8, false);

    const_generic_extra(1i8, 2u32, 42i32);
    const_generic_extra(true, 2u32, 42i32);
}
