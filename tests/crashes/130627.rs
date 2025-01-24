//@ known-bug: #130627

#![feature(trait_alias)]

trait Test {}

#[diagnostic::on_unimplemented(
    message="message",
    label="label",
    note="note"
)]
trait Alias = Test;

// Use trait alias as bound on type parameter.
fn foo<T: Alias>(v: &T) {
}

pub fn main() {
    foo(&1);
}
