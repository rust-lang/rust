//! Test for some of the existing limitations and the current error messages.
//! Some of these limitations may be removed in the future.

#![expect(incomplete_features)]
#![feature(contracts)]
#![allow(dead_code)]

/// Represent a 5-star system.
struct Stars(u8);

impl Stars {
    fn is_valid(&self) -> bool {
        self.0 <= 5
    }
}

trait ParseStars {
    #[core::contracts::ensures(|ret| ret.is_none_or(Stars::is_valid))]
    //~^ ERROR contract annotations is only supported in functions with bodies
    fn parse_string(input: String) -> Option<Stars>;

    #[core::contracts::ensures(|ret| ret.is_none_or(Stars::is_valid))]
    //~^ ERROR contract annotations is only supported in functions with bodies
    fn parse<T>(input: T) -> Option<Stars> where T: for<'a> Into<&'a str>;
}

fn main() {
}
