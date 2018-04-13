#![feature(plugin, custom_attribute)]
#![warn(excessive_precision)]
#![allow(print_literal)]

fn main() {
    // TODO add prefix tests
    // Consts
    const GOOD32_SUF: f32 = 0.123_456_f32;
    const GOOD32: f32 = 0.123_456;
    const GOOD32_SM: f32 = 0.000_000_000_1;
    const GOOD64: f64 = 0.123_456_789_012;
    const GOOD64_SM: f32 = 0.000_000_000_000_000_1;

    const BAD32_1: f32 = 0.123_456_789_f32;
    const BAD32_2: f32 = 0.123_456_789;
    const BAD32_3: f32 = 0.100_000_000_000_1;

    const BAD64_1: f64 = 0.123_456_789_012_345_67f64;
    const BAD64_2: f64 = 0.123_456_789_012_345_67;
    const BAD64_3: f64 = 0.100_000_000_000_000_000_1;

    // Literal
    println!("{}", 8.888_888_888_888_888_888_888);

    // TODO add inferred type tests for f32
    // TODO add tests cases exactly on the edge
    // Locals
    let good32: f32 = 0.123_456_f32;
    let good32_2: f32 = 0.123_456;

    let good64: f64 = 0.123_456_789_012f64;
    let good64: f64 = 0.123_456_789_012;
    let good64_2 = 0.123_456_789_012;

    let bad32_1: f32 = 1.123_456_789_f32;
    let bad32_2: f32 = 1.123_456_789;

    let bad64_1: f64 = 0.123_456_789_012_345_67f64;
    let bad64_2: f64 = 0.123_456_789_012_345_67;
    let bad64_3 = 0.123_456_789_012_345_67;

    // TODO Vectors / nested vectors
    let vec32: Vec<f32> = vec![0.123_456_789];
    let vec64: Vec<f64> = vec![0.123_456_789_123_456_789];

    // Exponential float notation
    let good_e32: f32 = 1e-10;
    let bad_e32: f32 = 1.123_456_788_888e-10;

    let good_bige32: f32 = 1E-10;
    let bad_bige32: f32 = 1.123_456_788_888E-10;
}
