#![feature(trait_alias)]

use std::any::Any;

trait Trait {}

trait WithType {
    type Ctx: ?Sized;
}

trait Alias<T> = Any where T: Trait;

// Note, we do not impl `Trait` for `()` here, so should expect the error below.

impl<T> WithType for T {
    type Ctx = dyn Alias<T>;
}

fn main() {
    let _: Box<<() as WithType>::Ctx> = Box::new(());
    //~^ ERROR the trait bound `(): Trait` is not satisfied [E0277]
}
