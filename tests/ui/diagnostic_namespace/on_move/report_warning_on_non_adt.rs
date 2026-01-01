#[diagnostic::on_move(
    message = "Foo",
    label = "Bar",
)]
struct Foo;

#[diagnostic::on_move(
//~^WARN `#[diagnostic::on_move]` can only be applied to enums, structs or unions
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
