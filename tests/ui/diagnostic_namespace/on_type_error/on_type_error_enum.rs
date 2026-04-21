#![feature(diagnostic_on_type_error)]

#[diagnostic::on_type_error(note = "expected enum `{Self}<{T}>`, found `{Found}`")]
enum MyEnum<T> {
    Variant(T),
}

fn main() {
    let e: MyEnum<&str> = "hello";
    //~^ ERROR mismatched types
    //~| NOTE expected enum `MyEnum<&str>`, found `&'static str
    //~| NOTE expected `MyEnum<&str>`, found `&str`
    //~| NOTE expected due to this
    //~| NOTE   expected enum `MyEnum<&str>`
}
