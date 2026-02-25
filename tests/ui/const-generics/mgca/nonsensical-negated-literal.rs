#![feature(adt_const_params, min_generic_const_args, unsized_const_params)]
#![expect(incomplete_features)]

use std::marker::ConstParamTy;

#[derive(Eq, PartialEq, ConstParamTy)]
struct Foo {
    field: isize
}

fn foo<const F: Foo>() {}

fn bar<const B: &'static str>() {}

fn main() {
    foo::<{ Foo { field: -1_usize } }>();
    //~^ ERROR: type annotations needed for the literal
    foo::<{ Foo { field: { -1_usize } } }>();
    //~^ ERROR: type annotations needed for the literal
    foo::<{ Foo { field: -true } }>();
    //~^ ERROR negated literal must be an integer
    foo::<{ Foo { field: { -true } } }>();
    //~^ ERROR negated literal must be an integer
    foo::<{ Foo { field: -"<3" } }>();
    //~^ ERROR negated literal must be an integer
    foo::<{ Foo { field: { -"<3" } } }>();
    //~^ ERROR negated literal must be an integer

    bar::<{ -"hi" }>();
    //~^ ERROR: negated literal must be an integer
}
