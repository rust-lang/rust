const ONE: i64 = 1;
const NEG_ONE: i64 = -1;
const ZERO: i64 = 0;

struct A(String);

impl std::ops::Shl<i32> for A {
    type Output = A;
    fn shl(mut self, other: i32) -> Self {
        self.0.push_str(&format!("{}", other));
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
}
