// run-pass
// Test that non-static methods can be assigned to local variables as
// function pointers.


trait Foo {
    fn foo(&self) -> usize;
}

struct A(usize);

impl A {
    fn bar(&self) -> usize { self.0 }
}

impl Foo for A {
    fn foo(&self) -> usize { self.bar() }
}

fn main() {
    let f = A::bar;
    let g = Foo::foo;
    let a = A(42);

    assert_eq!(f(&a), g(&a));
}
