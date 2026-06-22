//! Test that the feature gate is required for diagnostic_on_type_error

#[diagnostic::on_type_error(note = "custom on_type_error note: expected {Expected}, found {Found}")]
//~^ WARN unknown diagnostic attribute
//~| NOTE `#[warn(unknown_diagnostic_attributes)]` (part of `#[warn(unknown_or_malformed_diagnostic_attributes)]`) on by default
#[derive(Debug)]
struct Foo<T>(T);

fn takes_foo(_: Foo<i32>) {}
//~^ NOTE function defined here

fn main() {
    let foo = Foo(String::new());
    takes_foo(foo);
    //~^ERROR mismatched types
    //~| NOTE arguments to this function are incorrect
    //~| NOTE expected `Foo<i32>`, found `Foo<String>`
    //~| NOTE expected struct `Foo<i32>`
}
