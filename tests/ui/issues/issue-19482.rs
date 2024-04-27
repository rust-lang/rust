// Test that a partially specified trait object with unspecified associated
// type does not type-check.

trait Foo {
    type A;

    fn dummy(&self) { }
}

fn bar(x: &dyn Foo) {}
//~^ ERROR the associated type `A` in `Foo` must be specified

pub fn main() {}
