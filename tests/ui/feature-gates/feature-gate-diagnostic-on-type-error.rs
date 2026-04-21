//! Test that the feature gate is required for diagnostic_on_type_error

#[diagnostic::on_type_error(note = "expected {Expected}, found {Found}")]
//~^ WARN unknown diagnostic attribute
#[derive(Debug)]
struct Foo<T>(T);

fn takes_foo(_: Foo<i32>) {}

fn main() {
    let foo = Foo(String::new());
    takes_foo(foo);
    //~^ERROR mismatched types
}
