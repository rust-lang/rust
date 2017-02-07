#![feature(plugin)]
#![plugin(clippy)]

#![deny(absurd_extreme_comparisons)]
#![allow(unused, eq_op, no_effect, unnecessary_operation)]

fn main() {
    const Z: u32 = 0;

    let u: u32 = 42;

    u <= 0;
    //~^ ERROR this comparison involving the minimum or maximum element for this type contains a
    //~| HELP using u == 0 instead
    u <= Z;
    //~^ ERROR this comparison involving
    //~| HELP using u == Z instead
    u < Z;
    //~^ ERROR this comparison involving
    //~| HELP comparison is always false
    Z >= u;
    //~^ ERROR this comparison involving
    //~| HELP using Z == u instead
    Z > u;
    //~^ ERROR this comparison involving
    //~| HELP comparison is always false
    u > std::u32::MAX;
    //~^ ERROR this comparison involving
    //~| HELP comparison is always false
    u >= std::u32::MAX;
    //~^ ERROR this comparison involving
    //~| HELP using u == std::u32::MAX instead
    std::u32::MAX < u;
    //~^ ERROR this comparison involving
    //~| HELP comparison is always false
    std::u32::MAX <= u;
    //~^ ERROR this comparison involving
    //~| HELP using std::u32::MAX == u instead

    1-1 > u;
        //~^ ERROR this comparison involving
        //~| HELP because 1-1 is the minimum value for this type, this comparison is always false
    u >= !0;
        //~^ ERROR this comparison involving
        //~| HELP consider using u == !0 instead
    u <= 12 - 2*6;
        //~^ ERROR this comparison involving
        //~| HELP consider using u == 12 - 2*6 instead

    let i: i8 = 0;
    i < -127 - 1;
    //~^ ERROR this comparison involving
    //~| HELP comparison is always false
    std::i8::MAX >= i;
    //~^ ERROR this comparison involving
    //~| HELP comparison is always true
    3-7 < std::i32::MIN;
    //~^ ERROR this comparison involving
    //~| HELP comparison is always false

    let b = false;
    b >= true;
    //~^ ERROR this comparison involving
    //~| HELP using b == true instead
    false > b;
    //~^ ERROR this comparison involving
    //~| HELP comparison is always false

    u > 0; // ok

    // this is handled by unit_cmp
    () < {}; //~WARNING <-comparison of unit values detected.
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
