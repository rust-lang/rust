#![feature(rustc_attrs)]

#[rustc_on_unimplemented(on(
    all(A = "{integer}", any(Self = "[{integral}; _]",)),
    message = "an array of type `{Self}` cannot be built directly from an iterator",
))]
pub trait FromIterator<A>: Sized {
    fn from_iter<T: IntoIterator<Item = A>>(iter: T) -> Self;
}
fn main() {
    let iter = 0..42_8;
    let x: [u8; 8] = FromIterator::from_iter(iter);
    //~^ ERROR an array of type `[u8; 8]` cannot be built directly from an iterator
}
