#![feature(diagnostic_on_type_error)]

#[diagnostic::on_type_error(note)]
//~^ WARN malformed `diagnostic::on_type_error` attribute
//~| NOTE `#[warn(malformed_diagnostic_attributes)]` (part of `#[warn(unknown_or_malformed_diagnostic_attributes)]`) on by default
//~| NOTE invalid option found here
struct NoValue<T>(T);

#[diagnostic::on_type_error(note =)]
//~^ WARN expected a literal or missing delimiter
struct EmptyValue<T>(T);

fn main() {
    // Create type errors
    let _: NoValue<i32> = 43;
    //~^ ERROR mismatched types
    //~| NOTE expected due to this
    //~| NOTE expected `NoValue<i32>`, found integer
    //~| NOTE expected struct `NoValue<i32>`
    let _: EmptyValue<i32> = 44;
    //~^ ERROR mismatched types
    //~| NOTE expected due to this
    //~| NOTE expected `EmptyValue<i32>`, found integer
    //~| NOTE expected struct `EmptyValue<i32>`
}
