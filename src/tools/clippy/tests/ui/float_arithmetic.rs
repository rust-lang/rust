#![warn(clippy::integer_arithmetic, clippy::float_arithmetic)]
#![allow(
    unused,
    clippy::shadow_reuse,
    clippy::shadow_unrelated,
    clippy::no_effect,
    clippy::unnecessary_operation,
    clippy::op_ref
)]

#[rustfmt::skip]
fn main() {
    let mut f = 1.0f32;

    f * 2.0;

    1.0 + f;
    f * 2.0;
    f / 2.0;
    f - 2.0 * 4.2;
    -f;

    f += 1.0;
    f -= 1.0;
    f *= 2.0;
    f /= 2.0;
}

// also warn about floating point arith with references involved

pub fn float_arith_ref() {
    3.1_f32 + &1.2_f32;
    &3.4_f32 + 1.5_f32;
    &3.5_f32 + &1.3_f32;
}

pub fn float_foo(f: &f32) -> f32 {
    let a = 5.1;
    a + f
}

pub fn float_bar(f1: &f32, f2: &f32) -> f32 {
    f1 + f2
}

pub fn float_baz(f1: f32, f2: &f32) -> f32 {
    f1 + f2
}

pub fn float_qux(f1: f32, f2: f32) -> f32 {
    (&f1 + &f2)
}
