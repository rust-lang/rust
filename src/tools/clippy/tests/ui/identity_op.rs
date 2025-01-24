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
    //~^ ERROR: this operation has no effect
    x + (1 - 1);
    //~^ ERROR: this operation has no effect
    x + 1;
    0 + x;
    //~^ ERROR: this operation has no effect
    1 + x;
    x - ZERO; //no error, as we skip lookups (for now)
    x | (0);
    //~^ ERROR: this operation has no effect
    ((ZERO)) | x; //no error, as we skip lookups (for now)

    x * 1;
    //~^ ERROR: this operation has no effect
    1 * x;
    //~^ ERROR: this operation has no effect
    x / ONE; //no error, as we skip lookups (for now)

    x / 2; //no false positive

    x & NEG_ONE; //no error, as we skip lookups (for now)
    -1 & x;
    //~^ ERROR: this operation has no effect

    let u: u8 = 0;
    u & 255;
    //~^ ERROR: this operation has no effect

    1 << 0; // no error, this case is allowed, see issue 3430
    42 << 0;
    //~^ ERROR: this operation has no effect
    1 >> 0;
    //~^ ERROR: this operation has no effect
    42 >> 0;
    //~^ ERROR: this operation has no effect
    &x >> 0;
    //~^ ERROR: this operation has no effect
    x >> &0;
    //~^ ERROR: this operation has no effect

    let mut a = A(String::new());
    let b = a << 0; // no error: non-integer

    1 * Meter; // no error: non-integer

    2 % 3;
    //~^ ERROR: this operation has no effect
    -2 % 3;
    //~^ ERROR: this operation has no effect
    2 % -3 + x;
    //~^ ERROR: this operation has no effect
    -2 % -3 + x;
    //~^ ERROR: this operation has no effect
    x + 1 % 3;
    //~^ ERROR: this operation has no effect
    (x + 1) % 3; // no error
    4 % 3; // no error
    4 % -3; // no error

    // See #8724
    let a = 0;
    let b = true;
    0 + if b { 1 } else { 2 };
    //~^ ERROR: this operation has no effect
    0 + if b { 1 } else { 2 } + if b { 3 } else { 4 };
    //~^ ERROR: this operation has no effect
    0 + match a { 0 => 10, _ => 20 };
    //~^ ERROR: this operation has no effect
    0 + match a { 0 => 10, _ => 20 } + match a { 0 => 30, _ => 40 };
    //~^ ERROR: this operation has no effect
    0 + if b { 1 } else { 2 } + match a { 0 => 30, _ => 40 };
    //~^ ERROR: this operation has no effect
    0 + match a { 0 => 10, _ => 20 } + if b { 3 } else { 4 };
    //~^ ERROR: this operation has no effect
    (if b { 1 } else { 2 }) + 0;
    //~^ ERROR: this operation has no effect

    0 + { a } + 3;
    //~^ ERROR: this operation has no effect
    0 + { a } * 2;
    //~^ ERROR: this operation has no effect
    0 + loop { let mut c = 0; if c == 10 { break c; } c += 1; } + { a * 2 };
    //~^ ERROR: this operation has no effect

    fn f(_: i32) {
        todo!();
    }

    f(1 * a + { 8 * 5 });
    //~^ ERROR: this operation has no effect
    f(0 + if b { 1 } else { 2 } + 3);
    //~^ ERROR: this operation has no effect

    const _: i32 = { 2 * 4 } + 0 + 3;
    //~^ ERROR: this operation has no effect
    const _: i32 = 0 + { 1 + 2 * 3 } + 3;
    //~^ ERROR: this operation has no effect

    0 + a as usize;
    //~^ ERROR: this operation has no effect
    let _ = 0 + a as usize;
    //~^ ERROR: this operation has no effect
    0 + { a } as usize;
    //~^ ERROR: this operation has no effect

    2 * (0 + { a });
    //~^ ERROR: this operation has no effect
    1 * ({ a } + 4);
    //~^ ERROR: this operation has no effect
    1 * 1;
    //~^ ERROR: this operation has no effect

    // Issue #9904
    let x = 0i32;
    let _: i32 = &x + 0;
    //~^ ERROR: this operation has no effect
}

pub fn decide(a: bool, b: bool) -> u32 {
    0 + if a { 1 } else { 2 } + if b { 3 } else { 5 }
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
        //~^ ERROR: this operation has no effect
        let _: i32 = x + 0;
        //~^ ERROR: this operation has no effect
    }
    {
        let x = &&0i32;
        let _: i32 = **x + 0;
        //~^ ERROR: this operation has no effect
        let x = &&0i32;
        let _: i32 = *x + 0;
        //~^ ERROR: this operation has no effect
    }
    {
        // this is just silly
        let x = &&&0i32;
        let _: i32 = ***x + 0;
        //~^ ERROR: this operation has no effect
        let _: i32 = **x + 0;
        //~^ ERROR: this operation has no effect
        let x = 0i32;
        let _: i32 = *&x + 0;
        //~^ ERROR: this operation has no effect
        let _: i32 = **&&x + 0;
        //~^ ERROR: this operation has no effect
        let _: i32 = *&*&x + 0;
        //~^ ERROR: this operation has no effect
        let _: i32 = **&&*&x + 0;
        //~^ ERROR: this operation has no effect
    }
    {
        // this is getting ridiculous, but we should still see the same
        // error message so let's just keep going
        let x = &0i32;
        let _: i32 = **&&*&x + 0;
        //~^ ERROR: this operation has no effect
        let _: i32 = **&&*&x + 0;
        //~^ ERROR: this operation has no effect
    }
}

fn issue_13470() {
    let x = 1i32;
    let y = 1i32;
    // Removes the + 0i32 while keeping the parentheses around x + y so the cast operation works
    let _: u64 = (x + y + 0i32) as u64;
    //~^ ERROR: this operation has no effect
    // both of the next two lines should look the same after rustfix
    let _: u64 = 1u64 & (x + y + 0i32) as u64;
    //~^ ERROR: this operation has no effect
    // Same as above, but with extra redundant parenthesis
    let _: u64 = 1u64 & ((x + y) + 0i32) as u64;
    //~^ ERROR: this operation has no effect
    // Should maintain parenthesis even if the surrounding expr has the same precedence
    let _: u64 = 5u64 + ((x + y) + 0i32) as u64;
    //~^ ERROR: this operation has no effect

    // If we don't maintain the parens here, the behavior changes
    let _ = -(x + y + 0i32);
    //~^ ERROR: this operation has no effect
    // Similarly, we need to maintain parens here
    let _ = -(x / y / 1i32);
    //~^ ERROR: this operation has no effect
    // Maintain parenthesis if the parent expr is of higher precedence
    let _ = 2i32 * (x + y + 0i32);
    //~^ ERROR: this operation has no effect
    // Maintain parenthesis if the parent expr is the same precedence
    // as not all operations are associative
    let _ = 2i32 - (x - y - 0i32);
    //~^ ERROR: this operation has no effect
    // But make sure that inner parens still exist
    let z = 1i32;
    let _ = 2 + (x + (y * z) + 0);
    //~^ ERROR: this operation has no effect
    // Maintain parenthesis if the parent expr is of lower precedence
    // This is for clarity, and clippy will not warn on these being unnecessary
    let _ = 2i32 + (x * y * 1i32);
    //~^ ERROR: this operation has no effect

    let x = 1i16;
    let y = 1i16;
    let _: u64 = 1u64 + ((x as i32 + y as i32) as u64 + 0u64);
    //~^ ERROR: this operation has no effect
}
