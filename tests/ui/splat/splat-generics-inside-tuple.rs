// tests/ui/splat/splat-generics-inside-tuple.rs

//@ run-pass
//! Test using `#[splat]` on tuples with generics inside the splatted tuple.
#![allow(incomplete_features)]
#![feature(splat)]

fn generic_second<T>(#[splat] _s: (u32, T)) {}

fn generic_first<T>(#[splat] _s: (T, u32)) {}

fn generic_both<T, U>(#[splat] _s: (T, U)) {}

fn generic_triple<T, U, V>(#[splat] _s: (T, U, V)) {}

fn main() {
    generic_second(1u32, 2i8);
    generic_second(1u32, 2.0f64);
    generic_second(1u32, "hello");

    generic_first(1i8, 2u32);
    generic_first(2.0f64, 2u32);
    generic_first("hello", 2u32);

    generic_both(1u32, 2i8);
    generic_both("hello", 2.0f64);
    generic_both(true, "world");

    generic_triple(1u32, 2.0f64, "hello");
    generic_triple(true, 42i32, 3.14f32);
}
