#![feature(diagnostic_on_type_error)]

#[diagnostic::on_type_error]
//~^ WARN missing options for `diagnostic::on_type_error` attribute
//~| NOTE `#[warn(malformed_diagnostic_attributes)]` (part of `#[warn(unknown_or_malformed_diagnostic_attributes)]`) on by default
struct MissingOptions<T>(T);

fn main() {
    // Create a type error
    let _: MissingOptions<i32> = 32;
    //~^ ERROR mismatched types
    //~| NOTE expected due to this
    //~| NOTE expected `MissingOptions<i32>`, found integer
    //~| NOTE expected struct `MissingOptions<i32>`
}
