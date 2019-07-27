// run-pass

// Ensure that provided items are inherited properly even when impls vary in
// type parameters *and* rely on projections.

#![feature(specialization)]

use std::convert::Into;

trait Trait {
    fn to_u8(&self) -> u8;
}
trait WithAssoc {
    type Item;
    fn to_item(&self) -> Self::Item;
}

impl<T, U> Trait for T where T: WithAssoc<Item=U>, U: Into<u8> {
    fn to_u8(&self) -> u8 {
        self.to_item().into()
    }
}

impl WithAssoc for u8 {
    type Item = u8;
    fn to_item(&self) -> u8 { *self }
}

impl Trait for u8 {}

fn main() {
    assert!(3u8.to_u8() == 3u8);
}
