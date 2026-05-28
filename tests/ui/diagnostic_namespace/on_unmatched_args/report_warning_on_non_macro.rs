//@ check-pass
#![feature(diagnostic_on_unmatched_args)]

#[diagnostic::on_unmatched_args(message = "not allowed here")]
//~^ WARN cannot be used on
struct Foo;

fn main() {
    let _ = Foo;
}
