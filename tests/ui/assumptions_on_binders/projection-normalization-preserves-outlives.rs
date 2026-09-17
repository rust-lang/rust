//@ check-fail
//@ compile-flags: -Znext-solver=globally -Zassumptions-on-binders

use std::any::Any;

struct Outlives<T: 'static>(Option<T>);

trait Trait {
    fn foo<T>(x: T) -> (Box<dyn Any>, impl Sized) {
        //~^ ERROR the parameter type `T` may not live long enough
        (Box::new(x), Outlives::<T>(None))
        //~^ ERROR the parameter type `T` may not live long enough
        //~| ERROR the parameter type `T` may not live long enough
        //~| ERROR the parameter type `T` may not live long enough
    }
}

fn main() {}
