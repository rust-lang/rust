#[diagnostic::on_move(
    //~^WARN unused attribute
    //|=
    message = "first message",
    label = "first label",
)]
#[diagnostic::on_move(
    message = "second message",
    label = "second label",
)]
struct Foo;

fn takes_foo(_: Foo) {}

fn main() {
    let foo = Foo;
    takes_foo(foo);
    let bar = foo;
    //~^ERROR first message
}
