#![feature(diagnostic_on_move)]

#[diagnostic::on_move(
    message = "Foo",
    label = "Bar",
)]
struct Foo;

#[diagnostic::on_move(
//~^WARN the `diagnostic::on_move` attribute cannot be used on traits
    message = "Foo",
    label = "Bar",
)]
trait MyTrait {}

fn takes_foo(_: Foo) {}

fn main() {
    let foo = Foo;
    takes_foo(foo);
    let bar = foo;
    //~^ERROR Foo
}
