#![warn(clippy::identity_op)]
#![allow(unused)]
#![allow(
    clippy::eq_op,
    clippy::no_effect,
    clippy::unnecessary_operation,
    clippy::op_ref,
    clippy::double_parens,
    clippy::uninlined_format_args,
    clippy::borrow_deref_ref,
    clippy::deref_addrof
)]

use std::fmt::Write as _;

const ONE: i64 = 1;
const NEG_ONE: i64 = -1;
const ZERO: i64 = 0;

struct A(String);

impl std::ops::Shl<i32> for A {
    type Output = A;
    fn shl(mut self, other: i32) -> Self {
        let _ = write!(self.0, "{}", other);
        self
    }
}

struct Length(u8);
struct Meter;

impl core::ops::Mul<Meter> for u8 {
    type Output = Length;
    fn mul(self, _: Meter) -> Length {
        Length(self)
    }
}

#[rustfmt::skip]
fn main() {
    let x = 0;

    x + 0;
    //~^ identity_op

    x + (1 - 1);
    //~^ identity_op

    x + 1;
    0 + x;
    //~^ identity_op

    1 + x;
    x - ZERO; //no error, as we skip lookups (for now)
    x | (0);
    //~^ identity_op

    ((ZERO)) | x; //no error, as we skip lookups (for now)

    x * 1;
    //~^ identity_op

    1 * x;
    //~^ identity_op

    x / ONE; //no error, as we skip lookups (for now)

    x / 2; //no false positive

    x & NEG_ONE; //no error, as we skip lookups (for now)
    -1 & x;
    //~^ identity_op


    let u: u8 = 0;
    u & 255;
    //~^ identity_op


    1 << 0; // no error, this case is allowed, see issue 3430
    42 << 0;
    //~^ identity_op

    1 >> 0;
    //~^ identity_op

    42 >> 0;
    //~^ identity_op

    &x >> 0;
    //~^ identity_op

    x >> &0;
    //~^ identity_op


    let mut a = A(String::new());
    let b = a << 0; // no error: non-integer

    1 * Meter; // no error: non-integer

    2 % 3;
    //~^ identity_op

    -2 % 3;
    //~^ identity_op

    2 % -3 + x;
    //~^ identity_op

    -2 % -3 + x;
    //~^ identity_op

    x + 1 % 3;
    //~^ identity_op

    (x + 1) % 3; // no error
    4 % 3; // no error
    4 % -3; // no error

    // See #8724
    let a = 0;
    let b = true;
    0 + if b { 1 } else { 2 };
    //~^ identity_op

    0 + if b { 1 } else { 2 } + if b { 3 } else { 4 };
    //~^ identity_op

    0 + match a { 0 => 10, _ => 20 };
    //~^ identity_op

    0 + match a { 0 => 10, _ => 20 } + match a { 0 => 30, _ => 40 };
    //~^ identity_op

    0 + if b { 1 } else { 2 } + match a { 0 => 30, _ => 40 };
    //~^ identity_op

    0 + match a { 0 => 10, _ => 20 } + if b { 3 } else { 4 };
    //~^ identity_op

    (if b { 1 } else { 2 }) + 0;
    //~^ identity_op


    0 + { a } + 3;
    //~^ identity_op

    0 + { a } * 2;
    //~^ identity_op

    0 + loop { let mut c = 0; if c == 10 { break c; } c += 1; } + { a * 2 };
    //~^ identity_op


    fn f(_: i32) {
        todo!();
    }

    f(1 * a + { 8 * 5 });
    //~^ identity_op

    f(0 + if b { 1 } else { 2 } + 3);
    //~^ identity_op


    const _: i32 = { 2 * 4 } + 0 + 3;
    //~^ identity_op

    const _: i32 = 0 + { 1 + 2 * 3 } + 3;
    //~^ identity_op


    0 + a as usize;
    //~^ identity_op

    let _ = 0 + a as usize;
    //~^ identity_op

    0 + { a } as usize;
    //~^ identity_op


    2 * (0 + { a });
    //~^ identity_op

    1 * ({ a } + 4);
    //~^ identity_op

    1 * 1;
    //~^ identity_op


    // Issue #9904
    let x = 0i32;
    let _: i32 = &x + 0;
    //~^ identity_op

}

pub fn decide(a: bool, b: bool) -> u32 {
    0 + if a { 1 } else { 2 } + if b { 3 } else { 5 }
    //~^ identity_op
}

/// The following tests are from / for issue #12050
/// In short, the lint didn't work for coerced references,
/// e.g. let x = &0; let y = x + 0;
/// because the suggested fix was `let y = x;` but
/// it should have been `let y = *x;`
fn issue_12050() {
    {
        let x = &0i32;
        let _: i32 = *x + 0;
        //~^ identity_op

        let _: i32 = x + 0;
        //~^ identity_op
    }
    {
        let x = &&0i32;
        let _: i32 = **x + 0;
        //~^ identity_op

        let x = &&0i32;
        let _: i32 = *x + 0;
        //~^ identity_op
    }
    {
        // this is just silly
        let x = &&&0i32;
        let _: i32 = ***x + 0;
        //~^ identity_op

        let _: i32 = **x + 0;
        //~^ identity_op

        let x = 0i32;
        let _: i32 = *&x + 0;
        //~^ identity_op

        let _: i32 = **&&x + 0;
        //~^ identity_op

        let _: i32 = *&*&x + 0;
        //~^ identity_op

        let _: i32 = **&&*&x + 0;
        //~^ identity_op
    }
    {
        // this is getting ridiculous, but we should still see the same
        // error message so let's just keep going
        let x = &0i32;
        let _: i32 = **&&*&x + 0;
        //~^ identity_op

        let _: i32 = **&&*&x + 0;
        //~^ identity_op
    }
}

fn issue_13470() {
    let x = 1i32;
    let y = 1i32;
    // Removes the + 0i32 while keeping the parentheses around x + y so the cast operation works
    let _: u64 = (x + y + 0i32) as u64;
    //~^ identity_op

    // both of the next two lines should look the same after rustfix
    let _: u64 = 1u64 & (x + y + 0i32) as u64;
    //~^ identity_op

    // Same as above, but with extra redundant parenthesis
    let _: u64 = 1u64 & ((x + y) + 0i32) as u64;
    //~^ identity_op

    // Should maintain parenthesis even if the surrounding expr has the same precedence
    let _: u64 = 5u64 + ((x + y) + 0i32) as u64;
    //~^ identity_op

    // If we don't maintain the parens here, the behavior changes
    let _ = -(x + y + 0i32);
    //~^ identity_op

    // Similarly, we need to maintain parens here
    let _ = -(x / y / 1i32);
    //~^ identity_op

    // Maintain parenthesis if the parent expr is of higher precedence
    let _ = 2i32 * (x + y + 0i32);
    //~^ identity_op

    // Maintain parenthesis if the parent expr is the same precedence
    // as not all operations are associative
    let _ = 2i32 - (x - y - 0i32);
    //~^ identity_op

    // But make sure that inner parens still exist
    let z = 1i32;
    let _ = 2 + (x + (y * z) + 0);
    //~^ identity_op

    // Maintain parenthesis if the parent expr is of lower precedence
    // This is for clarity, and clippy will not warn on these being unnecessary
    let _ = 2i32 + (x * y * 1i32);
    //~^ identity_op

    let x = 1i16;
    let y = 1i16;
    let _: u64 = 1u64 + ((x as i32 + y as i32) as u64 + 0u64);
    //~^ identity_op
}

fn issue_14932() {
    let _ = 0usize + &Default::default(); // no error

    0usize + &Default::default(); // no error

    0usize + &<usize as Default>::default();
    //~^ identity_op

    let _ = 0usize + &usize::default();
    //~^ identity_op

    let _n: usize = 0usize + &Default::default();
    //~^ identity_op
}

// Expr's type can be inferred by the function's return type
fn issue_14932_2() -> usize {
    0usize + &Default::default()
    //~^ identity_op
}

trait Def {
    fn def() -> Self;
}

impl Def for usize {
    fn def() -> Self {
        0
    }
}

fn issue_14932_3() {
    let _ = 0usize + &Def::def(); // no error

    0usize + &Def::def(); // no error

    0usize + &<usize as Def>::def();
    //~^ identity_op

    let _ = 0usize + &usize::def();
    //~^ identity_op

    let _n: usize = 0usize + &Def::def();
    //~^ identity_op
}
