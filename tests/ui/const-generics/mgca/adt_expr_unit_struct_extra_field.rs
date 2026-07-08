#![feature(min_generic_const_args, adt_const_params)]

#[derive(Eq, PartialEq, std::marker::ConstParamTy)]
struct Foo;

fn foo<const N: Foo>() {}

fn main() {
    foo::<{ Foo { field: const { 1 } } }>();
    //~^ ERROR struct `Foo` has no field named `field` [E0560]
}
