//@ reference: attributes.diagnostic.on_unimplemented.intro
//@ reference: attributes.diagnostic.on_unimplemented.keys
#[diagnostic::on_unimplemented(message = "Foo", label = "Bar", note = "Baz")]
trait Foo {}

fn takes_foo(_: impl Foo) {}

fn main() {
    takes_foo(());
    //~^ERROR Foo
}
