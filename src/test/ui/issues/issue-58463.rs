// run-pass
// compile-flags:-C debuginfo=2
fn foo() -> impl Copy {
    foo
}
fn main() {
    foo();
}
