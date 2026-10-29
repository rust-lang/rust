use std::fmt::Debug;

trait Foo {
    fn foo(&self, _: &impl Debug);
}

impl Foo for () {
    fn foo<U: Debug>(&self, _: &U) { }
    //~^ ERROR method `foo` has incompatible signature for trait
}

trait Bar {
    fn bar<U: Debug>(&self, _: &U);
}

impl Bar for () {
    fn bar(&self, _: &impl Debug) { }
    //~^ ERROR method `bar` has incompatible signature for trait
}

trait Baz {
    fn baz<U: Debug, T: Debug>(&self, _: &U, _: &T);
}

impl Baz for () {
    fn baz<T: Debug>(&self, _: &impl Debug, _: &T) { }
    //~^ ERROR method `baz` has incompatible signature for trait
}

// With non-local trait (#49841):

use std::hash::{Hash, Hasher};

struct X;

impl Hash for X {
    fn hash(&self, hasher: &mut impl Hasher) {}
    //~^ ERROR method `hash` has incompatible signature for trait
}

fn main() {}
