// Formerly this ICEd with the following message:
// Tried to project an inherited associated type during coherence checking,
// which is currently not supported.
//
// No we expect to run into a more user-friendly cycle error instead.

// revisions: old re

#![cfg_attr(re, feature(re_rebalance_coherence))]
#![feature(specialization)]

trait Trait<T> { type Assoc; }
//[old]~^ cycle detected
//[re]~^^ ERROR E0391

impl<T> Trait<T> for Vec<T> {
    type Assoc = ();
}

impl Trait<u8> for Vec<u8> {}

impl<T> Trait<T> for String {
    type Assoc = ();
}

impl Trait<<Vec<u8> as Trait<u8>>::Assoc> for String {}

fn main() {}
