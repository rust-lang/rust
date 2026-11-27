#![feature(type_alias_impl_trait)]

use std::fmt::Debug;

type Foo = impl Debug;

fn is_send<T: Send>() {}

fn not_good() {
    // This function does not define `Foo`,
    // so it can actually check auto traits on the hidden type without
    // risking cycle errors.
    is_send::<Foo>();
}

#[define_opaque(Foo)]
fn not_gooder() -> Foo {
    // Constrain `Foo = u32`
    let x: Foo = 22_u32;

    // while we could know this from the hidden type, it would
    // need extra roundabout logic to support it.
    is_send::<Foo>();
    //~^ ERROR: type annotations needed: cannot satisfy `Foo: Send`

    x
}

fn main() {}
