#![feature(diagnostic_on_type_error)]

#[diagnostic::on_type_error(
    note = "custom on_type_error note: expected enum `{This}<{T}>`, found `{Found}`"
)]
enum MyEnum<T> {
    Variant(T),
}

fn main() {
    let e: MyEnum<&str> = "hello";
    //~^ ERROR mismatched types
    //~| NOTE expected `MyEnum<&str>`, found `&str`
    //~| NOTE custom on_type_error note: expected enum `MyEnum<&str>`, found `&'static str
    //~| NOTE expected due to this
    //~| NOTE   expected enum `MyEnum<&str>`
}
