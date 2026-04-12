//! This is an unusual feature gate test, as it doesn't test the feature
//! gate, but the fact that not adding the feature gate will cause the
//! diagnostic to not emit the custom diagnostic message
//!
#[diagnostic::on_type_error(note = "expected `{Expected}`, found `{Found}`")]
//~^ WARN unknown diagnostic attribute
#[derive(Debug)]
struct Foo<T>(T);

fn takes_foo(_: Foo<i32>) {}

fn main() {
    let foo: Foo<String> = Foo(42);
    //~^ ERROR mismatched types
    takes_foo(foo);
    //~^ ERROR mismatched types
    let bar: Foo<i32> = Foo("");
    //~^ ERROR mismatched types
}
