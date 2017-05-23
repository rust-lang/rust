#![feature(plugin)]
#![plugin(clippy)]

#![warn(absurd_extreme_comparisons)]
#![allow(unused, eq_op, no_effect, unnecessary_operation, needless_pass_by_value)]

fn main() {
    const Z: u32 = 0;
    let u: u32 = 42;
    u <= 0;
    u <= Z;
    u < Z;
    Z >= u;
    Z > u;
    u > std::u32::MAX;
    u >= std::u32::MAX;
    std::u32::MAX < u;
    std::u32::MAX <= u;
    1-1 > u;
    u >= !0;
    u <= 12 - 2*6;
    let i: i8 = 0;
    i < -127 - 1;
    std::i8::MAX >= i;
    3-7 < std::i32::MIN;
    let b = false;
    b >= true;
    false > b;
    u > 0; // ok
    // this is handled by unit_cmp
    () < {};
}

use std::cmp::{Ordering, PartialEq, PartialOrd};

#[derive(PartialEq, PartialOrd)]
pub struct U(u64);

impl PartialEq<u32> for U {
    fn eq(&self, other: &u32) -> bool {
        self.eq(&U(*other as u64))
    }
}
impl PartialOrd<u32> for U {
    fn partial_cmp(&self, other: &u32) -> Option<Ordering> {
        self.partial_cmp(&U(*other as u64))
    }
}

pub fn foo(val: U) -> bool {
    val > std::u32::MAX
}
