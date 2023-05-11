// run-pass
// Test that when you use ufcs form to invoke a trait method (on a
// trait object) everything works fine.


trait Foo {
    fn test(&self) -> i32;
}

impl Foo for i32 {
    fn test(&self) -> i32 { *self }
}

fn main() {
    let a: &dyn Foo = &22;
    assert_eq!(Foo::test(a), 22);
}
