#![feature(diagnostic_on_type_error)]

#[diagnostic::on_type_error(
    note = "custom on_type_error note: expected `{Expected}`, found `{Found}`"
)]
struct Array<T, const N: usize>([T; N]);
//~^ WARN `#[diagnostic::on_type_error]` only supports exactly one ADT generic parameter, but found `2` [malformed_diagnostic_attributes]
//~| NOTE `#[warn(malformed_diagnostic_attributes)]` (part of `#[warn(unknown_or_malformed_diagnostic_attributes)]`) on by default

struct OtherArray<T, const N: usize>([T; N]);

fn main() {
    let other: OtherArray<i32, 4> = OtherArray([1, 2, 3, 4]);

    let _: Array<i32, 4> = other;
    //~^ ERROR mismatched types
    //~| NOTE expected due to this
    //~| NOTE expected `Array<i32, 4>`, found `OtherArray<i32, 4>`
    //~| NOTE custom on_type_error note: expected `Array<i32, 4>`, found `OtherArray<i32, 4>`
    //~| NOTE expected struct `Array<i32, 4>`
}
