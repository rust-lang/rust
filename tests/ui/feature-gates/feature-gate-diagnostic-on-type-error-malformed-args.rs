//@ check-pass

// Regression test for https://github.com/rust-lang/rust/issues/158628.

#[diagnostic::on_type_error(unknown = "")]
//~^ WARN unknown diagnostic attribute
pub struct Foo {}

fn main() {}
