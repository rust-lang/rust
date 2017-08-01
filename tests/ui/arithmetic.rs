#![feature(plugin)]
#![plugin(clippy)]

#![warn(integer_arithmetic, float_arithmetic)]
#![allow(unused, shadow_reuse, shadow_unrelated, no_effect, unnecessary_operation)]
fn main() {
    let i = 1i32;
    1 + i;
    i * 2;
    1 %
    i / 2; // no error, this is part of the expression in the preceding line
    i - 2 + 2 - i;
    -i;

    i & 1; // no wrapping
    i | 1;
    i ^ 1;
    i >> 1;
    i << 1;

    let f = 1.0f32;

    f * 2.0;

    1.0 + f;
    f * 2.0;
    f / 2.0;
    f - 2.0 * 4.2;
    -f;
}
