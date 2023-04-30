// run-pass
fn foo() -> impl Copy {
    foo
}
fn main() {
    foo();
}
