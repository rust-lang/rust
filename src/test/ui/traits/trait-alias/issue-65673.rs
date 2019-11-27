// run-pass

#![feature(trait_alias)]

use std::any::Any;

trait Trait {}

trait WithType {
    type Ctx: ?Sized;
}

trait Alias<T> = Any where T: Trait;

impl Trait for () {}

impl<T> WithType for T {
    type Ctx = dyn Alias<T>;
}

fn main() {
    let _: Box<<() as WithType>::Ctx> = Box::new(());
}
