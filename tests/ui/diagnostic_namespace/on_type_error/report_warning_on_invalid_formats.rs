#![feature(diagnostic_on_type_error)]

#[diagnostic::on_type_error(note = "custom on_type_error note: invalid format {Expected:123}")]
//~^ WARN format specifiers are not permitted in diagnostic attributes
//~| NOTE `#[warn(malformed_diagnostic_format_literals)]` (part of `#[warn(unknown_or_malformed_diagnostic_attributes)]`) on by default
//~| NOTE remove this format specifier
struct InvalidFormat1<T>(T);

#[diagnostic::on_type_error(note = "custom on_type_error note: invalid format {Expected:!}")]
//~^ WARN format specifiers are not permitted in diagnostic attributes
//~| NOTE remove this format specifier
struct InvalidFormat2<T>(T);

fn main() {
    // Create type errors to trigger the notes
    let _: InvalidFormat1<i32> = InvalidFormat2("");
    //~^ ERROR mismatched types
    //~| NOTE expected due to this
    //~| NOTE expected `InvalidFormat1<i32>`, found `InvalidFormat2<&str>`
    //~| NOTE custom on_type_error note: invalid format InvalidFormat1<i32>
    //~| NOTE expected struct `InvalidFormat1<i32>`
    let _: InvalidFormat2<i32> = InvalidFormat1(3.14);
    //~^ ERROR mismatched types
    //~| NOTE expected due to this
    //~| NOTE expected `InvalidFormat2<i32>`, found `InvalidFormat1<{float}>`
    //~| NOTE custom on_type_error note: invalid format InvalidFormat2<i32>
    //~| NOTE expected struct `InvalidFormat2<i32>`
}
