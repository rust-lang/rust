use std::hash::Hash;

use crate::fmt::Debug;

pub trait Interned<I>: Copy + Debug + Hash + Eq + PartialEq {
    type Value;
    fn get(self) -> Self::Value;
}
