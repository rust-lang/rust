#![feature(diagnostic_on_type_error)]

#[diagnostic::on_type_error]
//~^ WARN missing options for `diagnostic::on_type_error` attribute
struct MissingOptions<T>(T);

fn main() {
    // Create a type error
    let _: MissingOptions<i32> = 32;
    //~^ ERROR mismatched types
}
