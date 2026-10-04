//@ reference: attributes.diagnostic.on_unimplemented.note-repetition
#[diagnostic::on_unimplemented(message = "Foo", label = "Bar", note = "Baz", note = "Boom")]
trait Foo {}

#[diagnostic::on_unimplemented(message = "Bar", label = "Foo", note = "Baz")]
#[diagnostic::on_unimplemented(note = "Baz2")]
trait Bar {}

fn takes_foo(_: impl Foo) {}
fn takes_bar(_: impl Bar) {}

fn main() {
    takes_foo(());
    //~^ERROR Foo
    takes_bar(());
    //~^ERROR Bar
}
