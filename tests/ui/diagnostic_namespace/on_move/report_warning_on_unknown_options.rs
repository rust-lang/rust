#[diagnostic::on_move(
    message = "Foo",
    label = "Bar",
    baz="Baz"
    //~^WARN unknown or malformed `on_move` attribute
    //~|HELP only `message` and `label` are allowed as options
)]
struct Foo;

fn takes_foo(_: Foo) {}

fn main() {
    let foo = Foo;
    takes_foo(foo);
    let bar = foo;
    //~^ERROR Foo
}
