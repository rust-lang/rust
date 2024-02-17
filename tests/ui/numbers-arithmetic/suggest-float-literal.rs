//@ run-rustfix

#![allow(dead_code)]

fn add_integer_to_f32(x: f32) -> f32 {
    x + 100 //~ ERROR cannot add `{integer}` to `f32`
}

fn add_integer_to_f64(x: f64) -> f64 {
    x + 100 //~ ERROR cannot add `{integer}` to `f64`
}

fn subtract_integer_from_f32(x: f32) -> f32 {
    x - 100 //~ ERROR cannot subtract `{integer}` from `f32`
}

fn subtract_integer_from_f64(x: f64) -> f64 {
    x - 100 //~ ERROR cannot subtract `{integer}` from `f64`
}

fn multiply_f32_by_integer(x: f32) -> f32 {
    x * 100 //~ ERROR cannot multiply `f32` by `{integer}`
}

fn multiply_f64_by_integer(x: f64) -> f64 {
    x * 100 //~ ERROR cannot multiply `f64` by `{integer}`
}

fn divide_f32_by_integer(x: f32) -> f32 {
    x / 100 //~ ERROR cannot divide `f32` by `{integer}`
}

fn divide_f64_by_integer(x: f64) -> f64 {
    x / 100 //~ ERROR cannot divide `f64` by `{integer}`
}

fn main() {}
