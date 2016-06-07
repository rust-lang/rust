#![feature(plugin)]
#![plugin(clippy)]

#![deny(absurd_extreme_comparisons)]
#![allow(unused, eq_op, no_effect, unnecessary_operation)]
fn main() {
    const Z: u32 = 0;

    let u: u32 = 42;

    u <= 0;
    //~^ ERROR this comparison involving the minimum or maximum element for this type contains a case that is always true or always false
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
        //~| HELP because !0 is the maximum value for this type, the case where the two sides are not equal never occurs, consider using u == !0 instead
    u <= 12 - 2*6;
        //~^ ERROR this comparison involving
        //~| HELP because 12 - 2*6 is the minimum value for this type, the case where the two sides are not equal never occurs, consider using u == 12 - 2*6 instead

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
