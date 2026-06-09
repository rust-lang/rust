// ignore-tidy-linelength
//@ revisions: bpass1 bfail2
//@ edition: 2021
//@ [bpass1] compile-flags: --crate-type lib --emit dep-info,metadata
//@ [bfail2] aux-build: circular-dependencies-aux.rs
//@ [bfail2] compile-flags: --test --extern aux={{build-base}}/circular-dependencies/auxiliary/libcircular_dependencies_aux.rmeta -L dependency={{build-base}}/circular-dependencies

pub struct Foo;
//[bfail2]~^ NOTE there are multiple different versions of crate `circular_dependencies` in the dependency graph
//[bfail2]~| NOTE there are multiple different versions of crate `circular_dependencies` in the dependency graph
//[bfail2]~| NOTE this is the expected type
//[bfail2]~| NOTE this is the expected type
//[bfail2]~| NOTE this is the found type
//[bfail2]~| NOTE this is the found type

pub fn consume_foo(_: Foo) {}
//[bfail2]~^ NOTE function defined here

pub fn produce_foo() -> Foo {
    Foo
}

#[test]
fn test() {
    aux::consume_foo(produce_foo());
    //[bfail2]~^ ERROR mismatched types [E0308]
    //[bfail2]~| NOTE expected `circular_dependencies::Foo`, found `Foo`
    //[bfail2]~| NOTE arguments to this function are incorrect
    //[bfail2]~| NOTE function defined here

    consume_foo(aux::produce_foo());
    //[bfail2]~^ ERROR mismatched types [E0308]
    //[bfail2]~| NOTE expected `Foo`, found `circular_dependencies::Foo`
    //[bfail2]~| NOTE arguments to this function are incorrect
}
