// Test that a partially specified trait object with unspecified associated
// type does not type-check.

trait Foo {
    type A;

    fn dummy(&self) { }
}

fn bar(x: &Foo) {}
//~^ ERROR the associated type `A` (from the trait `Foo`) must be specified

pub fn main() {}
