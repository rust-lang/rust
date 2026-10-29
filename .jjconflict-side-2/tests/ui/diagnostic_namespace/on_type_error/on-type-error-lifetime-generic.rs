#![feature(diagnostic_on_type_error)]
#[diagnostic::on_type_error(
    note = "custom on_type_error note: expected `{Expected}`, found `{Found}`"
)]
struct Ref<'a, T>(&'a T);

struct OtherRef<'a, T>(&'a T);

fn main() {
    let value = 42;

    let other: OtherRef<'_, i32> = OtherRef(&value);

    let _: Ref<'_, i32> = other;
    //~^ ERROR mismatched types
    //~| NOTE expected struct `Ref<'_, i32>`
    //~| NOTE expected due to this
    //~| NOTE expected `Ref<'_, i32>`, found `OtherRef<'_, i32>`
    //~| NOTE custom on_type_error note: expected `Ref<'_, i32>`, found `OtherRef<'_, i32>`
}
