//@ reference: attributes.diagnostic.on_unimplemented.repetition
//@ reference: attributes.diagnostic.on_unimplemented.syntax
#[diagnostic::on_unimplemented(
    if(Self = "()"),
    //~^WARN malformed `on_unimplemented` attribute
    //~|WARN malformed `on_unimplemented` attribute
    message = "custom message",
    note = "custom note"
)]
#[diagnostic::on_unimplemented(message = "fallback!!")]
//~^ WARN `message` is ignored due to previous definition of `message`
//~| WARN `message` is ignored due to previous definition of `message`
#[diagnostic::on_unimplemented(label = "fallback label")]
#[diagnostic::on_unimplemented(note = "fallback note")]
trait Foo {}

fn takes_foo(_: impl Foo) {}

fn main() {
    takes_foo(());
    //~^ERROR custom message
}
