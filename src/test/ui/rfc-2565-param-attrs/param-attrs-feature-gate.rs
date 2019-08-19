// gate-test-param_attrs

#![deny(unused_variables)]

fn foo(
    /// Foo
    //~^ ERROR documentation comments cannot be applied to function parameters
    //~| NOTE doc comments are not allowed here
    //~| ERROR attributes on function parameters are unstable
    //~| NOTE https://github.com/rust-lang/rust/issues/60406
    #[allow(unused_variables)] a: u8
    //~^ ERROR attributes on function parameters are unstable
    //~| NOTE https://github.com/rust-lang/rust/issues/60406
) {}

fn main() {}
