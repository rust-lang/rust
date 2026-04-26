//@ check-pass
#![feature(diagnostic_on_unmatch_args)]

#[diagnostic::on_unmatch_args(message = "not allowed here")]
//~^ WARN `#[diagnostic::on_unmatch_args]` can only be applied to macro definitions
struct Foo;

fn main() {
    let _ = Foo;
}
