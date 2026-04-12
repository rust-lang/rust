#![feature(diagnostic_on_type_error)]

#[diagnostic::on_type_error(unknown = "option")]
//~^ WARN malformed `diagnostic::on_type_error` attribute
struct UnknownOption<T>(T);

#[diagnostic::on_type_error(message = "not allowed")]
//~^ WARN malformed `diagnostic::on_type_error` attribute
struct MessageOption<T>(T);

#[diagnostic::on_type_error(label = "not allowed")]
//~^ WARN malformed `diagnostic::on_type_error` attribute
struct LabelOption<T>(T);

fn main() {
    // Create type errors
    let _: UnknownOption<i32> = MessageOption(43);
    //~^ ERROR mismatched types
    let _: MessageOption<i32> = UnknownOption(32);
    //~^ ERROR mismatched types
    let _: LabelOption<i32> = UnknownOption(32);
    //~^ ERROR mismatched types
}
