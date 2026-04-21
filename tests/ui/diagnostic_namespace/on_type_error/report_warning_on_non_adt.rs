#![feature(diagnostic_on_type_error)]

#[diagnostic::on_type_error(note = "not an ADT")]
//~^ WARN `#[diagnostic::on_type_error]` can only be applied to enums, structs or unions
fn function() {}

#[diagnostic::on_type_error(note = "not an ADT")]
//~^ WARN `#[diagnostic::on_type_error]` can only be applied to enums, structs or unions
static STATIC: i32 = 0;

#[diagnostic::on_type_error(note = "not an ADT")]
//~^ WARN `#[diagnostic::on_type_error]` can only be applied to enums, structs or unions
mod module {}

#[diagnostic::on_type_error(note = "not an ADT")]
//~^ WARN `#[diagnostic::on_type_error]` can only be applied to enums, structs or unions
trait Trait {}

#[diagnostic::on_type_error(note = "not an ADT")]
//~^ WARN `#[diagnostic::on_type_error]` can only be applied to enums, structs or unions
type TypeAlias = i32;

struct SomeStruct;

impl SomeStruct {
    #[diagnostic::on_type_error(note = "not an ADT")]
    //~^ WARN `#[diagnostic::on_type_error]` can only be applied to enums, structs or unions
    fn method() {}
}

fn main() {
    // Create a type error with a valid ADT to ensure the feature works
    #[diagnostic::on_type_error(note = "expected `{Expected}`, found `{Found}`")]
    struct Valid<T>(T);

    let _: Valid<i32> = SomeStruct;
    //~^ ERROR mismatched types
}
