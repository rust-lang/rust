#![warn(clippy::arithmetic_side_effects, clippy::float_arithmetic)]
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
    //~^ float_arithmetic



    1.0 + f;
    //~^ float_arithmetic

    f * 2.0;
    //~^ float_arithmetic

    f / 2.0;
    //~^ float_arithmetic

    f - 2.0 * 4.2;
    //~^ float_arithmetic

    -f;
    //~^ float_arithmetic


    f += 1.0;
    //~^ float_arithmetic

    f -= 1.0;
    //~^ float_arithmetic

    f *= 2.0;
    //~^ float_arithmetic

    f /= 2.0;
    //~^ float_arithmetic

}

// also warn about floating point arith with references involved

pub fn float_arith_ref() {
    3.1_f32 + &1.2_f32;
    //~^ float_arithmetic

    &3.4_f32 + 1.5_f32;
    //~^ float_arithmetic

    &3.5_f32 + &1.3_f32;
    //~^ float_arithmetic
}

pub fn float_foo(f: &f32) -> f32 {
    let a = 5.1;
    a + f
    //~^ float_arithmetic
}

pub fn float_bar(f1: &f32, f2: &f32) -> f32 {
    f1 + f2
    //~^ float_arithmetic
}

pub fn float_baz(f1: f32, f2: &f32) -> f32 {
    f1 + f2
    //~^ float_arithmetic
}

pub fn float_qux(f1: f32, f2: f32) -> f32 {
    (&f1 + &f2)
    //~^ float_arithmetic
}
