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

#[allow(
    clippy::eq_op,
    clippy::no_effect,
    clippy::unnecessary_operation,
    clippy::op_ref,
    clippy::double_parens
)]
#[warn(clippy::identity_op)]
#[rustfmt::skip]
fn main() {
    let x = 0;

    x + 0;
    x + (1 - 1);
    x + 1;
    0 + x;
    1 + x;
    x - ZERO; //no error, as we skip lookups (for now)
    x | (0);
    ((ZERO)) | x; //no error, as we skip lookups (for now)

    x * 1;
    1 * x;
    x / ONE; //no error, as we skip lookups (for now)

    x / 2; //no false positive

    x & NEG_ONE; //no error, as we skip lookups (for now)
    -1 & x;

    let u: u8 = 0;
    u & 255;

    1 << 0; // no error, this case is allowed, see issue 3430
    42 << 0;
    1 >> 0;
    42 >> 0;
    &x >> 0;
    x >> &0;

    let mut a = A("".into());
    let b = a << 0; // no error: non-integer

    1 * Meter; // no error: non-integer

    2 % 3;
    -2 % 3;
    2 % -3 + x;
    -2 % -3 + x;
    x + 1 % 3;
    (x + 1) % 3; // no error
    4 % 3; // no error
    4 % -3; // no error

    // See #8724
    let a = 0;
    let b = true;
    0 + if b { 1 } else { 2 };
    0 + if b { 1 } else { 2 } + if b { 3 } else { 4 }; // no error
    0 + match a { 0 => 10, _ => 20 };
    0 + match a { 0 => 10, _ => 20 } + match a { 0 => 30, _ => 40 }; // no error
    0 + if b { 1 } else { 2 } + match a { 0 => 30, _ => 40 }; // no error
    0 + match a { 0 => 10, _ => 20 } + if b { 3 } else { 4 }; // no error
    
    0 + if b { 0 + 1 } else { 2 };
    0 + match a { 0 =>  0 + 10, _ => 20 };
    0 + if b { 0 + 1 } else { 2 } + match a { 0 => 0 + 30, _ => 40 };

    let _ = 0 + if 0 + 1 > 0 { 1 } else { 2 } + if 0 + 1 > 0 { 3 } else { 4 };
    let _ = 0 + match 0 + 1 { 0 => 10, _ => 20 } + match 0 + 1  { 0 => 30, _ => 40 };

    0 + if b { 1 } else { 2 } + if b { 3 } else { 4 } + 0;
    
    0 + { a } + 3; // no error
    0 + loop { let mut c = 0; if c == 10 { break c; } c += 1; } + { a * 2 }; // no error
    
    fn f(_: i32) {
        todo!();
    }
    f(1 * a + { 8 * 5 });
    f(0 + if b { 1 } else { 2 } + 3); // no error
    const _: i32 = { 2 * 4 } + 0 + 3;
    const _: i32 = 0 + { 1 + 2 * 3 } + 3; // no error
}
