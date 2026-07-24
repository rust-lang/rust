//@ run-pass
//! Test using `#[arg_splat]` on tuples with generics inside the splatted tuple.
#![allow(incomplete_features)]
#![feature(arg_splat)]

fn generic_second<T>(#[arg_splat] _s: (u32, T)) {}

fn generic_first<T>(#[arg_splat] _s: (T, u32)) {}

fn generic_both<T, U>(#[arg_splat] _s: (T, U)) {}

fn generic_triple<T, U, V>(#[arg_splat] _s: (T, U, V)) {}

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
