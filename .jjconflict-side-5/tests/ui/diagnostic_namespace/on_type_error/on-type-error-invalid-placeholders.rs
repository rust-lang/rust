#![feature(diagnostic_on_move)]
#[diagnostic::on_move(message = "Foo", label = "Bar", note = "{Expected} {Found}")]
//~^ WARN unknown parameter `Expected` [malformed_diagnostic_format_literals]
//~| WARN unknown parameter `Found` [malformed_diagnostic_format_literals]
#[derive(Debug)]
struct Foo;
fn takes_foo(_: Foo) {}
fn main() {
    let foo = Foo;
    takes_foo(foo);
    let bar = foo;
    //~^ERROR Foo
}
