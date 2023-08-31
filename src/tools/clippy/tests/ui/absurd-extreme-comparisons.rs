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
    //~^ ERROR: this comparison involving the minimum or maximum element for this type con
    u <= Z;
    //~^ ERROR: this comparison involving the minimum or maximum element for this type con
    u < Z;
    //~^ ERROR: this comparison involving the minimum or maximum element for this type con
    Z >= u;
    //~^ ERROR: this comparison involving the minimum or maximum element for this type con
    Z > u;
    //~^ ERROR: this comparison involving the minimum or maximum element for this type con
    u > u32::MAX;
    //~^ ERROR: this comparison involving the minimum or maximum element for this type con
    u >= u32::MAX;
    //~^ ERROR: this comparison involving the minimum or maximum element for this type con
    u32::MAX < u;
    //~^ ERROR: this comparison involving the minimum or maximum element for this type con
    u32::MAX <= u;
    //~^ ERROR: this comparison involving the minimum or maximum element for this type con
    1-1 > u;
    //~^ ERROR: this comparison involving the minimum or maximum element for this type con
    u >= !0;
    //~^ ERROR: this comparison involving the minimum or maximum element for this type con
    u <= 12 - 2*6;
    //~^ ERROR: this comparison involving the minimum or maximum element for this type con
    let i: i8 = 0;
    i < -127 - 1;
    //~^ ERROR: this comparison involving the minimum or maximum element for this type con
    i8::MAX >= i;
    //~^ ERROR: this comparison involving the minimum or maximum element for this type con
    3-7 < i32::MIN;
    //~^ ERROR: this comparison involving the minimum or maximum element for this type con
    let b = false;
    b >= true;
    //~^ ERROR: this comparison involving the minimum or maximum element for this type con
    false > b;
    //~^ ERROR: this comparison involving the minimum or maximum element for this type con
    u > 0; // ok
    // this is handled by clippy::unit_cmp
    () < {};
    //~^ ERROR: <-comparison of unit values detected. This will always be false
    //~| NOTE: `#[deny(clippy::unit_cmp)]` on by default
}

use std::cmp::{Ordering, PartialEq, PartialOrd};

#[derive(PartialEq, Eq, PartialOrd)]
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
