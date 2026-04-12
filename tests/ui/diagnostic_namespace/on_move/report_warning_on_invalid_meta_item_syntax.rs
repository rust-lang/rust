#![feature(diagnostic_on_move)]

#[diagnostic::on_move = "foo"]
//~^WARN missing options for `diagnostic::on_move` attribute [malformed_diagnostic_attributes]
struct Foo;

fn takes_foo(_: Foo) {}

fn main() {
    let foo = Foo;
    takes_foo(foo);
    let bar = foo;
    //~^ERROR use of moved value: `foo`
}
