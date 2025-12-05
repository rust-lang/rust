//@ check-pass

// Make sure we don't crash with a cycle error during coherence.

#![feature(specialization)] //~ WARN the feature `specialization` is incomplete

trait Trait<T> {
    type Assoc;
}

impl<T> Trait<T> for Vec<T> {
    default type Assoc = ();
}

impl Trait<u8> for Vec<u8> {
    type Assoc = u8;
}

impl<T> Trait<T> for String {
    type Assoc = ();
}

impl Trait<<Vec<u8> as Trait<u8>>::Assoc> for String {}

fn main() {}
