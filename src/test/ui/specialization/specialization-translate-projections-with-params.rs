// run-pass

// Ensure that provided items are inherited properly even when impls vary in
// type parameters *and* rely on projections, and the type parameters are input
// types on the trait.

#![feature(specialization)]

trait Trait<T> {
    fn convert(&self) -> T;
}
trait WithAssoc {
    type Item;
    fn as_item(&self) -> &Self::Item;
}

impl<T, U> Trait<U> for T where T: WithAssoc<Item=U>, U: Clone {
    fn convert(&self) -> U {
        self.as_item().clone()
    }
}

impl WithAssoc for u8 {
    type Item = u8;
    fn as_item(&self) -> &u8 { self }
}

impl Trait<u8> for u8 {}

fn main() {
    assert!(3u8.convert() == 3u8);
}
