#[diagnostic::on_move(
    message = "Foo for {Self}",
    label = "Bar for {Self}",
)]
#[derive(Debug)]
struct Foo;

fn takes_foo(_: Foo) {}

fn main() {
    let foo = Foo;
    takes_foo(foo);
    let bar = foo;
    //~^ERROR Foo for Foo
}
