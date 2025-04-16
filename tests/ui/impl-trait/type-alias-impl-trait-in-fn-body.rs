//! Test that it is basically not possible to declare *and opaquely use* opaque types
//! in function bodies. This will work again once we have a `#[defines]` attribute

#![feature(type_alias_impl_trait)]

use std::fmt::Debug;

fn main() {
    //~^ ERROR: item does not constrain
    type Existential = impl Debug;

    #[define_opaque(Existential)]
    fn f() -> Existential {}
    println!("{:?}", f());
}
