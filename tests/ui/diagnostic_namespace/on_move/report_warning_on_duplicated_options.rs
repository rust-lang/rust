#![feature(diagnostic_on_move)]

#[diagnostic::on_move(
    message = "first message",
    label = "first label",
)]
#[diagnostic::on_move(
    message = "second message",
    //~^ WARN `message` is ignored due to previous definition of `message` [malformed_diagnostic_attributes]
    label = "second label",
    //~^ WARN `label` is ignored due to previous definition of `label` [malformed_diagnostic_attributes]
)]
struct Foo;

fn takes_foo(_: Foo) {}

fn main() {
    let foo = Foo;
    takes_foo(foo);
    let bar = foo;
    //~^ERROR first message
}
