#![feature(diagnostic_on_type_error)]

#[diagnostic::on_type_error(note)]
//~^ WARN malformed `diagnostic::on_type_error` attribute
struct NoValue<T>(T);

#[diagnostic::on_type_error(note =)]
//~^ WARN expected a literal or missing delimiter
struct EmptyValue<T>(T);

fn main() {
    // Create type errors
    let _: NoValue<i32> = 43;
    //~^ ERROR mismatched types
    let _: EmptyValue<i32> = 44;
    //~^ ERROR mismatched types
}
