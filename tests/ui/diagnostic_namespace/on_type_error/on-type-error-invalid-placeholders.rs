#![feature(diagnostic_on_move)]
#[diagnostic::on_move(message = "Foo", label = "Bar", note = "{expected} {found}")]
//~^ WARN unknown parameter `expected` [malformed_diagnostic_format_literals]
//~| WARN unknown parameter `found` [malformed_diagnostic_format_literals]
#[derive(Debug)]
struct Foo;
fn takes_foo(_: Foo) {}
fn main() {
    let foo = Foo;
    takes_foo(foo);
    let bar = foo;
    //~^ERROR Foo
}
