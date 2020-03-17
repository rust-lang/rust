#![warn(clippy::integer_arithmetic, clippy::float_arithmetic)]
#![allow(
    unused,
    clippy::shadow_reuse,
    clippy::shadow_unrelated,
    clippy::no_effect,
    clippy::unnecessary_operation,
    clippy::op_ref,
    clippy::trivially_copy_pass_by_ref
)]

#[rustfmt::skip]
fn main() {
    let mut i = 1i32;
    1 + i;
    i * 2;
    1 %
    i / 2; // no error, this is part of the expression in the preceding line
    i - 2 + 2 - i;
    -i;

    // no error, overflows are checked by `overflowing_literals`
    -1;
    -(-1);

    i & 1; // no wrapping
    i | 1;
    i ^ 1;
    i >> 1;
    i << 1;

    i += 1;
    i -= 1;
    i *= 2;
    i /= 2;
    i %= 2;

    // no errors
    i <<= 3;
    i >>= 2;
    i |= 1;
    i &= 1;
    i ^= i;

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

    // no error, overflows are checked by `overflowing_literals`
    -1.;
    -(-1.);

    // No errors for the following items because they are constant expressions
    enum Foo {
        Bar = -2,
    }
    struct Baz([i32; 1 + 1]);
    union Qux {
        field: [i32; 1 + 1],
    }
    type Alias = [i32; 1 + 1];

    const FOO: i32 = -2;
    static BAR: i32 = -2;

    let _: [i32; 1 + 1] = [0, 0];

    let _: [i32; 1 + 1] = {
        let a: [i32; 1 + 1] = [0, 0];
        a
    };

    trait Trait {
        const ASSOC: i32 = 1 + 1;
    }

    impl Trait for Foo {
        const ASSOC: i32 = {
            let _: [i32; 1 + 1];
            fn foo() {}
            1 + 1
        };
    }


}

// warn on references as well! (#5328)
pub fn int_arith_ref() {
    3 + &1;
    &3 + 1;
    &3 + &1;
}

pub fn foo(x: &i32) -> i32 {
    let a = 5;
    a + x
}

pub fn bar(x: &i32, y: &i32) -> i32 {
    x + y
}

pub fn baz(x: i32, y: &i32) -> i32 {
    x + y
}

pub fn qux(x: i32, y: i32) -> i32 {
    (&x + &y)
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
