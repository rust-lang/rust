//@ reference: attributes.diagnostic.on_unimplemented.repetition
#[diagnostic::on_unimplemented(
    message = "first message",
    label = "first label",
    note = "custom note"
)]
#[diagnostic::on_unimplemented(
    message = "second message",
    //~^WARN `message` is ignored due to previous definition of `message`
    //~|WARN `message` is ignored due to previous definition of `message`
    label = "second label",
    //~^WARN `label` is ignored due to previous definition of `label`
    //~|WARN `label` is ignored due to previous definition of `label`
    note = "second note"
)]
trait Foo {}


fn takes_foo(_: impl Foo) {}

fn main() {
    takes_foo(());
    //~^ERROR first message
}
