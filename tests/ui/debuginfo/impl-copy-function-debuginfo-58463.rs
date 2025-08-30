// https://github.com/rust-lang/rust/issues/58463
//@ run-pass
//@ compile-flags:-C debuginfo=2

fn foo() -> impl Copy {
    foo
}
fn main() {
    foo();
}
