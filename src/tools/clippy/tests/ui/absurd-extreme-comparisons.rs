#![warn(clippy::absurd_extreme_comparisons)]
#![allow(
    unused,
    clippy::eq_op,
    clippy::no_effect,
    clippy::unnecessary_operation,
    clippy::needless_pass_by_value
)]

#[rustfmt::skip]
fn main() {
    const Z: u32 = 0;
    let u: u32 = 42;
    u <= 0;
    u <= Z;
    u < Z;
    Z >= u;
    Z > u;
    u > u32::MAX;
    u >= u32::MAX;
    u32::MAX < u;
    u32::MAX <= u;
    1-1 > u;
    u >= !0;
    u <= 12 - 2*6;
    let i: i8 = 0;
    i < -127 - 1;
    i8::MAX >= i;
    3-7 < i32::MIN;
    let b = false;
    b >= true;
    false > b;
    u > 0; // ok
    // this is handled by clippy::unit_cmp
    () < {};
}

use std::cmp::{Ordering, PartialEq, PartialOrd};

#[derive(PartialEq, PartialOrd)]
pub struct U(u64);

impl PartialEq<u32> for U {
    fn eq(&self, other: &u32) -> bool {
        self.eq(&U(u64::from(*other)))
    }
}
impl PartialOrd<u32> for U {
    fn partial_cmp(&self, other: &u32) -> Option<Ordering> {
        self.partial_cmp(&U(u64::from(*other)))
    }
}

pub fn foo(val: U) -> bool {
    val > u32::MAX
}

pub fn bar(len: u64) -> bool {
    // This is OK as we are casting from target sized to fixed size
    len >= usize::MAX as u64
}
