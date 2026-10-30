// FIXME investigate and fix the ICE?

#![feature(checked_type_aliases, pin_ergonomics)]
#![allow(incomplete_features)]

#[pin_v2]
pub struct UnpinLocal<T> {
    pub foo: T
}

pub type UnpinLocalAlias<T> = UnpinLocal<T>;
impl <T> Unpin for UnpinLocalAlias<T> {}

fn main() {}
