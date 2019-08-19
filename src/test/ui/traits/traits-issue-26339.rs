// run-pass
// Test that the right implementation is called through a trait
// object when supertraits include multiple references to the
// same trait, with different type parameters.

trait A: PartialEq<Foo> + PartialEq<Bar> { }

struct Foo;
struct Bar;

struct Aimpl;

impl PartialEq<Foo> for Aimpl {
    fn eq(&self, _rhs: &Foo) -> bool {
        true
    }
}

impl PartialEq<Bar> for Aimpl {
    fn eq(&self, _rhs: &Bar) -> bool {
        false
    }
}

impl A for Aimpl { }

fn main() {
    let a = &Aimpl as &dyn A;

    assert!(*a == Foo);
}
