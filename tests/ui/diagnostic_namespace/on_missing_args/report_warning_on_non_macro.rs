//@ check-pass
#![feature(diagnostic_on_missing_args)]

#[diagnostic::on_missing_args(message = "not allowed here")]
//~^ WARN `#[diagnostic::on_missing_args]` can only be applied to macro definitions
struct Foo;

fn main() {
    let _ = Foo;
}
