#![feature(min_generic_const_args, adt_const_params)]
#![expect(incomplete_features)]

use std::marker::ConstParamTy;

#[derive(Eq, PartialEq, ConstParamTy)]
struct Point(u32, u32);

fn with_point<const P: Point>() {}

fn test<const N: u32>() {
    with_point::<{ Point(N + 1, N) }>();
    //~^ ERROR complex const arguments must be placed inside of a `const` block

    with_point::<{ Point(const { N + 1 }, N) }>();
    //~^ ERROR generic parameters may not be used in const operations
}

fn main() {}
