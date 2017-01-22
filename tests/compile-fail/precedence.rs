#![feature(plugin)]
#![plugin(clippy)]

#[deny(precedence)]
#[allow(identity_op)]
#[allow(eq_op)]
fn main() {
    1 << 2 + 3;
    //~^ ERROR operator precedence can trip
    //~| SUGGESTION 1 << (2 + 3)
    1 + 2 << 3;
    //~^ERROR operator precedence can trip
    //~| SUGGESTION (1 + 2) << 3
    4 >> 1 + 1;
    //~^ERROR operator precedence can trip
    //~| SUGGESTION 4 >> (1 + 1)
    1 + 3 >> 2;
    //~^ERROR operator precedence can trip
    //~| SUGGESTION (1 + 3) >> 2
    1 ^ 1 - 1;
    //~^ERROR operator precedence can trip
    //~| SUGGESTION 1 ^ (1 - 1)
    3 | 2 - 1;
    //~^ERROR operator precedence can trip
    //~| SUGGESTION 3 | (2 - 1)
    3 & 5 - 2;
    //~^ERROR operator precedence can trip
    //~| SUGGESTION 3 & (5 - 2)

    -1i32.abs();
    //~^ERROR unary minus has lower precedence
    //~| SUGGESTION -(1i32.abs())
    -1f32.abs();
    //~^ERROR unary minus has lower precedence
    //~| SUGGESTION -(1f32.abs())

    // These should not trigger an error
    let _ = (-1i32).abs();
    let _ = (-1f32).abs();
    let _ = -(1i32).abs();
    let _ = -(1f32).abs();
    let _ = -(1i32.abs());
    let _ = -(1f32.abs());
}
