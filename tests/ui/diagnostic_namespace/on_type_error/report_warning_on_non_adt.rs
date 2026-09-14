#![feature(diagnostic_on_type_error)]

#[diagnostic::on_type_error(note = "custom on_type_error note: not an ADT")]
//~^ WARN cannot be used on
fn function() {}

#[diagnostic::on_type_error(note = "custom on_type_error note: not an ADT")]
//~^ WARN cannot be used on
static STATIC: i32 = 0;

#[diagnostic::on_type_error(note = "custom on_type_error note: not an ADT")]
//~^ WARN cannot be used on
mod module {}

#[diagnostic::on_type_error(note = "custom on_type_error note: not an ADT")]
//~^ WARN cannot be used on
trait Trait {}

#[diagnostic::on_type_error(note = "custom on_type_error note: not an ADT")]
//~^ WARN cannot be used on
type TypeAlias = i32;

struct SomeStruct;

impl SomeStruct {
    #[diagnostic::on_type_error(note = "custom on_type_error note: not an ADT")]
    //~^ WARN cannot be used on
    fn method() {}
}

fn main() {
    // Create a type error with a valid ADT to ensure the feature works
    #[diagnostic::on_type_error(
        note = "custom on_type_error note: expected `{Expected}`, found `{Found}`"
    )]
    struct Valid<T>(T);

    let _: Valid<i32> = SomeStruct;
    //~^ ERROR mismatched types
}
