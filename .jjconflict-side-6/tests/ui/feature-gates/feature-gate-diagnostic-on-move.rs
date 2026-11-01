//! This is an unusual feature gate test, as it doesn't test the feature
//! gate, but the fact that not adding the feature gate will cause the
//! diagnostic to not emit the custom diagnostic message
//!
#[diagnostic::on_move(message = "Foo")]
//~^ WARN unknown diagnostic attribute
#[derive(Debug)]
struct Foo;

fn takes_foo(_: Foo) {}

fn main() {
    let foo = Foo;
    takes_foo(foo);
    let bar = foo;
    //~^ERROR use of moved value: `foo`
}
