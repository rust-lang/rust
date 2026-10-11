use std::hash::Hash;

pub trait Interned<I>: Copy + Hash + Eq + PartialEq {
    type Value;
    fn get(self) -> Self::Value;
}
