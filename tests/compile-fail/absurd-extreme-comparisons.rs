#![feature(plugin)]
#![plugin(clippy)]

#![deny(absurd_extreme_comparisons)]
#![allow(unused, eq_op, no_effect, unnecessary_operation)]
fn main() {
    const Z: u32 = 0;

    let u: u32 = 42;

    u <= 0; //~ERROR this comparison involving the minimum or maximum element for this type contains a case that is always true or always false
    u <= Z; //~ERROR this comparison involving
    u < Z; //~ERROR this comparison involving
    Z >= u; //~ERROR this comparison involving
    Z > u; //~ERROR this comparison involving
    u > std::u32::MAX; //~ERROR this comparison involving
    u >= std::u32::MAX; //~ERROR this comparison involving
    std::u32::MAX < u; //~ERROR this comparison involving
    std::u32::MAX <= u; //~ERROR this comparison involving

    1-1 > u;
        //~^ ERROR this comparison involving
        //~| HELP because 1-1 is the minimum value for this type, this comparison is always false
    u >= !0;
        //~^ ERROR this comparison involving
        //~| HELP because !0 is the maximum value for this type, the case where the two sides are not equal never occurs, consider using u == !0 instead
    u <= 12 - 2*6;
        //~^ ERROR this comparison involving
        //~| HELP because 12 - 2*6 is the minimum value for this type, the case where the two sides are not equal never occurs, consider using u == 12 - 2*6 instead

    let i: i8 = 0;
    i < -127 - 1; //~ERROR this comparison involving
    std::i8::MAX >= i; //~ERROR this comparison involving
    3-7 < std::i32::MIN; //~ERROR this comparison involving

    let b = false;
    b >= true; //~ERROR this comparison involving
    false > b; //~ERROR this comparison involving

    u > 0; // ok

    // this is handled by unit_cmp
    () < {}; //~WARNING <-comparison of unit values detected.
}
