#[allow(unused_assignments)]
#[warn(clippy::misrefactored_assign_op, clippy::assign_op_pattern)]
fn main() {
    let mut a = 5;
    a += a + 1;
    a += 1 + a;
    a -= a - 1;
    a *= a * 99;
    a *= 42 * a;
    a /= a / 2;
    a %= a % 5;
    a &= a & 1;
    a *= a * a;
    a = a * a * a;
    a = a * 42 * a;
    a = a * 2 + a;
    a -= 1 - a;
    a /= 5 / a;
    a %= 42 % a;
    a <<= 6 << a;
}

// check that we don't lint on op assign impls, because that's just the way to impl them

use std::ops::{Mul, MulAssign};

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Wrap(i64);

impl Mul<i64> for Wrap {
    type Output = Self;

    fn mul(self, rhs: i64) -> Self {
        Wrap(self.0 * rhs)
    }
}

impl MulAssign<i64> for Wrap {
    fn mul_assign(&mut self, rhs: i64) {
        *self = *self * rhs
    }
}

fn cow_add_assign() {
    use std::borrow::Cow;
    let mut buf = Cow::Owned(String::from("bar"));
    let cows = Cow::Borrowed("foo");

    // this can be linted
    buf = buf + cows.clone();

    // this should not as cow<str> Add is not commutative
    buf = cows + buf;
    println!("{}", buf);
}
