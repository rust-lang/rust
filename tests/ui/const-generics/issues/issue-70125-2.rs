//@ run-pass
fn main() {
    <()>::foo();
}

trait Foo<const X: usize> {
    fn foo() -> usize {
        X
    }
}

impl Foo<3> for () {}
