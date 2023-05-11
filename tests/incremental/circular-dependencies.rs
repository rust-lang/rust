// ignore-tidy-linelength
// revisions: cpass1 cfail2
// edition: 2021
// [cpass1] compile-flags: --crate-type lib --emit dep-info,metadata
// [cfail2] aux-build: circular-dependencies-aux.rs
// [cfail2] compile-flags: --test --extern aux={{build-base}}/circular-dependencies/auxiliary/libcircular_dependencies_aux.rmeta -L dependency={{build-base}}/circular-dependencies

pub struct Foo;
//[cfail2]~^ NOTE `Foo` is defined in the current crate
//[cfail2]~| NOTE `Foo` is defined in the current crate
//[cfail2]~| NOTE `circular_dependencies::Foo` is defined in crate `circular_dependencies`
//[cfail2]~| NOTE `circular_dependencies::Foo` is defined in crate `circular_dependencies`

pub fn consume_foo(_: Foo) {}
//[cfail2]~^ NOTE function defined here

pub fn produce_foo() -> Foo {
    Foo
}

#[test]
fn test() {
    aux::consume_foo(produce_foo());
    //[cfail2]~^ ERROR mismatched types [E0308]
    //[cfail2]~| NOTE expected `circular_dependencies::Foo`, found `Foo`
    //[cfail2]~| NOTE arguments to this function are incorrect
    //[cfail2]~| NOTE `Foo` and `circular_dependencies::Foo` have similar names, but are actually distinct types
    //[cfail2]~| NOTE the crate `circular_dependencies` is compiled multiple times, possibly with different configurations
    //[cfail2]~| NOTE function defined here

    consume_foo(aux::produce_foo());
    //[cfail2]~^ ERROR mismatched types [E0308]
    //[cfail2]~| NOTE expected `Foo`, found `circular_dependencies::Foo`
    //[cfail2]~| NOTE arguments to this function are incorrect
    //[cfail2]~| NOTE `circular_dependencies::Foo` and `Foo` have similar names, but are actually distinct types
    //[cfail2]~| NOTE the crate `circular_dependencies` is compiled multiple times, possibly with different configurations
}
