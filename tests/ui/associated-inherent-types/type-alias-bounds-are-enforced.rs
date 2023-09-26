// compile-flags: --crate-type=lib
// check-pass

#![feature(inherent_associated_types)]
#![allow(incomplete_features)]

// Bounds on the self type play a major role in the resolution of inherent associated types (*).
// As a result of that, if a type alias contains any then its bounds have to be respected and the
// lint `type_alias_bounds` should not fire.

#![deny(type_alias_bounds)]

pub type Alias<T: Bound> = (Source<T>::Assoc,);

pub struct Source<T>(T);
pub trait Bound {}

impl<T: Bound> Source<T> {
    pub type Assoc = ();
}
