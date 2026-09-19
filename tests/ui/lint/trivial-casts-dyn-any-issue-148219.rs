//! Trait-object upcasts used as method receivers can affect method lookup.

//@ check-pass

#![warn(trivial_casts)]

use std::any::Any;

trait DynKey: Any {}

impl<T: Any> DynKey for T {}

fn method_receiver(other: &dyn DynKey) {
    let _ = (other as &dyn Any).downcast_ref::<u32>();
}

fn plain_binding(other: &dyn DynKey) {
    // This cast is trivial, but it is not used as a method receiver,
    // so it should be linted.
    let _ = other as &dyn Any;
    //~^ WARN trivial cast
}

fn main() {
    method_receiver(&0u32);
    plain_binding(&0u32);
}
