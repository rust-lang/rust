#![feature(diagnostic_on_type_error)]

#[diagnostic::on_type_error(note = "too many generics")]
struct TooMany<T, U>(T, U);
//~^ WARN `#[diagnostic::on_type_error]` only supports one ADT generic parameter, but found `2`

fn main() {
    let _: TooMany<i32, i32> = TooMany(32, "test");
    //~^ ERROR mismatched types
}
