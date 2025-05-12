#![feature(impl_trait_in_assoc_type)]

trait Foo {
    type Foo;
    fn bar();
}

impl Foo for () {
    type Foo = impl std::fmt::Debug;
    //~^ ERROR: unconstrained opaque type
    fn bar() {
        let x: Self::Foo = ();
        //~^ ERROR: mismatched types
    }
}

fn main() {}
