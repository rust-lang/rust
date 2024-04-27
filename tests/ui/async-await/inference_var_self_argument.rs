//! This is a regression test for an ICE.
//@ edition: 2021

trait Foo {
    async fn foo(self: &dyn Foo) {
        //~^ ERROR: `Foo` cannot be made into an object
        //~| ERROR invalid `self` parameter type: &dyn Foo
        todo!()
    }
}

fn main() {}
