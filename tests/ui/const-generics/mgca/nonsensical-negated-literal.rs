#![feature(adt_const_params, min_generic_const_args)]
#![expect(incomplete_features)]

use std::marker::ConstParamTy;

#[derive(Eq, PartialEq, ConstParamTy)]
struct Foo {
    field: isize
}

fn foo<const F: Foo>() {}

fn main() {
    foo::<{ Foo { field: -1_usize } }>();
    //~^ ERROR: type annotations needed for the literal
    foo::<{ Foo { field: { -1_usize } } }>();
    //~^ ERROR: complex const arguments must be placed inside of a `const` block
    foo::<{ Foo { field: -true } }>();
    //~^ ERROR: the constant `true` is not of type `isize`
    foo::<{ Foo { field: { -true } } }>();
    //~^ ERROR: complex const arguments must be placed inside of a `const` block
    foo::<{ Foo { field: -"<3" } }>();
    //~^ ERROR: the constant `"<3"` is not of type `isize`
    foo::<{ Foo { field: { -"<3" } } }>();
    //~^ ERROR: complex const arguments must be placed inside of a `const` block
}
