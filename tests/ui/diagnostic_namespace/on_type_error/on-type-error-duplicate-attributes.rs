#![feature(diagnostic_on_type_error)]

#[diagnostic::on_type_error(note = "custom on_type_error note: test coalesce first")]
#[diagnostic::on_type_error(note = "custom on_type_error note: test coalesce second")]
struct S<T>(T);

struct K<T>(T);

fn main() {
    let k = K(42);

    let _: S<i32> = k;
    //~^ ERROR mismatched types
    //~| NOTE custom on_type_error note: test coalesce first
    //~| NOTE custom on_type_error note: test coalesce second
    //~| NOTE expected struct `S<i32>`
    //~| NOTE expected due to this
    //~| NOTE expected `S<i32>`, found `K<{integer}>`
}
