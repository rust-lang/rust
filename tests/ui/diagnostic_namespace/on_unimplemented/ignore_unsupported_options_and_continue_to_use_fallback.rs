#![feature(diagnostic_namespace)]

#[diagnostic::on_unimplemented(
    //~^WARN malformed `on_unimplemented` attribute
    //~|WARN malformed `on_unimplemented` attribute
    if(Self = ()),
    message = "not used yet",
    label = "not used yet",
    note = "not used yet"
)]
#[diagnostic::on_unimplemented(message = "fallback!!")]
#[diagnostic::on_unimplemented(label = "fallback label")]
#[diagnostic::on_unimplemented(note = "fallback note")]
#[diagnostic::on_unimplemented(message = "fallback2!!")]
trait Foo {}

fn takes_foo(_: impl Foo) {}

fn main() {
    takes_foo(());
    //~^ERROR fallback!!
}
