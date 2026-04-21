#![feature(diagnostic_on_type_error)]

#[diagnostic::on_type_error(note = "invalid format {Expected:123}")]
//~^ WARN invalid format specifier
struct InvalidFormat1<T>(T);

#[diagnostic::on_type_error(note = "invalid format {Expected:!}")]
//~^ WARN invalid format specifier
struct InvalidFormat2<T>(T);

fn main() {
    // Create type errors to trigger the notes
    let _: InvalidFormat1<i32> = InvalidFormat2("");
    //~^ ERROR mismatched types
    let _: InvalidFormat2<i32> = InvalidFormat1(3.14);
    //~^ ERROR mismatched types
}
