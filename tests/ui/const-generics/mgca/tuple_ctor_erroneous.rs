#![feature(min_generic_const_args, adt_const_params)]
#![expect(incomplete_features)]

use std::marker::ConstParamTy;

#[derive(Eq, PartialEq, ConstParamTy)]
struct Point(u32, u32);

#[derive(Eq, PartialEq, ConstParamTy)]
enum MyEnum<T> {
    Variant(T),
    Unit,
}

const CONST_ITEM: u32 = 42;

fn accepts_point<const P: Point>() {}
fn accepts_enum<const E: MyEnum<u32>>() {}

fn non_ctor() {}

fn test_errors<const N: usize>() {
    accepts_point::<{ Point(N) }>();
    //~^ ERROR tuple constructor has 2 arguments but 1 were provided

    accepts_point::<{ Point(N, N, N) }>();
    //~^ ERROR tuple constructor has 2 arguments but 3 were provided

    accepts_point::<{ UnresolvedIdent(N, N) }>();
    //~^ ERROR cannot find function, tuple struct or tuple variant `UnresolvedIdent` in this scope
    //~| ERROR tuple constructor with invalid base path

    accepts_point::<{ non_ctor(N, N) }>();
    //~^ ERROR tuple constructor with invalid base path

    accepts_point::<{ CONST_ITEM(N, N) }>();
    //~^ ERROR tuple constructor with invalid base path

    accepts_point::<{ Point }>();
    //~^ ERROR the constant `Point` is not of type `Point`

    accepts_enum::<{ MyEnum::Variant::<u32> }>();
    //~^ ERROR the constant `MyEnum::<u32>::Variant` is not of type `MyEnum<u32>`
}

fn main() {}
