// Formerly this ICEd with the following message:
// Tried to project an inherited associated type during coherence checking,
// which is currently not supported.
//
// No we expect to run into a more user-friendly cycle error instead.
#![feature(specialization)]
//~^ WARN the feature `specialization` is incomplete

trait Trait<T> { type Assoc; }
//~^ ERROR E0391

impl<T> Trait<T> for Vec<T> {
    type Assoc = ();
}

impl Trait<u8> for Vec<u8> {}

impl<T> Trait<T> for String {
    type Assoc = ();
}

impl Trait<<Vec<u8> as Trait<u8>>::Assoc> for String {}

fn main() {}
