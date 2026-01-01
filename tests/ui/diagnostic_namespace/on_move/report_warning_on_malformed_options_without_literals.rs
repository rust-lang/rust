#[diagnostic::on_move(
//~^ WARN expected a literal or missing delimiter [malformed_diagnostic_attributes]
//~| HELP only literals are allowed as values of `message` and `label`, separated by a comma
    message = Foo,
    label = "Bar",
)]
struct Foo;

fn takes_foo(_: Foo) {}

fn main() {
    let foo = Foo;
    takes_foo(foo);
    let bar = foo;
    //~^ERROR use of moved value: `foo`
}
