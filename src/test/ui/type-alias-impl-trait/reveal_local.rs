#![feature(type_alias_impl_trait)]

use std::fmt::Debug;

type Foo = impl Debug;
//~^ ERROR cycle detected
//~| ERROR cycle detected

fn is_send<T: Send>() { }

fn not_good() {
    // Error: this function does not constrain `Foo` to any particular
    // hidden type, so it cannot rely on `Send` being true.
    is_send::<Foo>();
}

fn not_gooder() {
    // Constrain `Foo = u32`
    let x: Foo = 22_u32;

    // while we could know this from the hidden type, it would
    // need extra roundabout logic to support it.
    is_send::<Foo>();
}

fn main() {}
