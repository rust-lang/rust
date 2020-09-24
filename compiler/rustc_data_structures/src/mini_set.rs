use crate::fx::FxHashSet;
use arrayvec::ArrayVec;
use std::hash::Hash;
/// Small-storage-optimized implementation of a set.
///
/// Stores elements in a small array up to a certain length
/// and switches to `HashSet` when that length is exceeded.
pub enum MiniSet<T> {
    Array(ArrayVec<[T; 8]>),
    Set(FxHashSet<T>),
}

impl<T: Eq + Hash> MiniSet<T> {
    /// Creates an empty `MiniSet`.
    pub fn new() -> Self {
        MiniSet::Array(ArrayVec::new())
    }

    /// Adds a value to the set.
    ///
    /// If the set did not have this value present, true is returned.
    ///
    /// If the set did have this value present, false is returned.
    pub fn insert(&mut self, elem: T) -> bool {
        match self {
            MiniSet::Array(array) => {
                if array.iter().any(|e| *e == elem) {
                    false
                } else {
                    if let Err(error) = array.try_push(elem) {
                        let mut set: FxHashSet<T> = array.drain(..).collect();
                        set.insert(error.element());
                        *self = MiniSet::Set(set);
                    }
                    true
                }
            }
            MiniSet::Set(set) => set.insert(elem),
        }
    }
}
