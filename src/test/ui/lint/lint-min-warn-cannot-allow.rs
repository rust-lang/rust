#![deny(warnings)]
//~^ NOTE lint level defined here

#![allow(illegal_floating_point_literal_pattern)]

fn _0() {
    if let 0.0 = 0.0 {}
    //~^ ERROR floating-point types cannot be used in patterns
    //~| NOTE #[deny(illegal_floating_point_literal_pattern)] implied by #[deny(warnings)]
    //~| NOTE #[warn(illegal_floating_point_literal_pattern)] is the minimum lint level
    //~| NOTE the lint level cannot be reduced to `allow`
    //~| WARN this was previously accepted
    //~| WARN hard error
    //~| NOTE for more information, see issue #41620
}

fn main() {}
