//@ check-pass
#![feature(diagnostic_on_unmatch_args)]

#[diagnostic::on_unmatched_args(message = "not allowed here")]
//~^ WARN `#[diagnostic::on_unmatched_args]` can only be applied to macro definitions
struct Foo;

fn main() {
    let _ = Foo;
}
