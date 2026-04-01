#![feature(diagnostic_on_move)]

#[diagnostic::on_move(
    message = "Foo",
    label = "Bar",
    baz="Baz"
    //~^WARN unknown or malformed `on_move` attribute
    //~|HELP only `message`, `note` and `label` are allowed as options. Their values must be string literals
)]
struct Foo;

fn takes_foo(_: Foo) {}

fn main() {
    let foo = Foo;
    takes_foo(foo);
    let bar = foo;
    //~^ERROR Foo
}
