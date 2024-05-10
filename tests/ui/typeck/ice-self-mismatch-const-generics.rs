// Checks that the following does not ICE when constructing type mismatch diagnostic involving
// `Self` and const generics.
// Issue: <https://github.com/rust-lang/rust/issues/122467>

pub struct GenericStruct<const N: usize, T> {
    thing: T,
}

impl<T> GenericStruct<0, T> {
    pub fn new(thing: T) -> GenericStruct<1, T> {
        Self { thing }
        //~^ ERROR mismatched types
    }
}

pub struct GenericStruct2<const M: usize, T>(T);

impl<T> GenericStruct2<0, T> {
    pub fn new(thing: T) -> GenericStruct2<1, T> {
        Self { 0: thing }
        //~^ ERROR mismatched types
    }
}

fn main() {}
