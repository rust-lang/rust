#![feature(diagnostic_on_type_error)]
#[diagnostic::on_type_error(
    note = "custom on_type_error note: expected struct `{Expected}`\n found struct `{Found}`"
)]
struct S<T>(T);
//~^ NOTE tuple struct defined here
struct K<T> {
    foo: T,
}
fn main() {
    let s: S<i32> = S(String::new());
    //~^ ERROR mismatched types
    //~| NOTE arguments to this struct are incorrect
    //~| NOTE expected `i32`, found `String`
    //~| NOTE this argument influences the type of `S`
    let k: K<i32> = K { foo: "" };
    //~^ ERROR mismatched types
    //~| NOTE expected `i32`, found `&str`
    let _: S<i32> = k;
    //~^ ERROR mismatched types
    //~| NOTE expected due to this
    //~| NOTE expected `S<i32>`, found `K<i32>`
    //~| NOTE custom on_type_error note: expected struct `S<i32>`
    //~| NOTE expected struct `S<i32>`
}
