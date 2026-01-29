#[diagnostic::on_move(
    message = "Foo {Baz}",
    //~^WARN unknown parameter `Baz`
    label = "Bar",
)]
#[derive(Debug)]
struct Foo;

fn takes_foo(_: Foo) {}

fn main() {
    let foo = Foo;
    takes_foo(foo);
    let bar = foo;
    //~^ERROR Foo
}
