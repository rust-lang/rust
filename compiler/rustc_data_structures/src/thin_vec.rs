use crate::stable_hasher::{HashStable, StableHasher};

use std::iter::FromIterator;

/// A vector type optimized for cases where this size is usually 0 (cf. `SmallVec`).
/// The `Option<Box<..>>` wrapping allows us to represent a zero sized vector with `None`,
/// which uses only a single (null) pointer.
#[derive(Clone, Encodable, Decodable, Debug, Hash, Eq, PartialEq)]
pub struct ThinVec<T>(Option<Box<Vec<T>>>);

impl<T> ThinVec<T> {
    pub fn new() -> Self {
        ThinVec(None)
    }

    pub fn iter(&self) -> std::slice::Iter<'_, T> {
        self.into_iter()
    }

    pub fn iter_mut(&mut self) -> std::slice::IterMut<'_, T> {
        self.into_iter()
    }

    pub fn push(&mut self, item: T) {
        match *self {
            ThinVec(Some(ref mut vec)) => vec.push(item),
            ThinVec(None) => *self = vec![item].into(),
        }
    }

    /// Note: if `set_len(0)` is called on a non-empty `ThinVec`, it will
    /// remain in the `Some` form. This is required for some code sequences
    /// (such as the one in `flat_map_in_place`) that call `set_len(0)` before
    /// an operation that might panic, and then call `set_len(n)` again
    /// afterwards.
    pub unsafe fn set_len(&mut self, new_len: usize) {
        match *self {
            ThinVec(None) => {
                // A prerequisite of `Vec::set_len` is that `new_len` must be
                // less than or equal to capacity(). The same applies here.
                if new_len != 0 {
                    panic!("unsafe ThinVec::set_len({})", new_len);
                }
            }
            ThinVec(Some(ref mut vec)) => vec.set_len(new_len),
        }
    }

    pub fn insert(&mut self, index: usize, value: T) {
        match *self {
            ThinVec(None) => {
                if index == 0 {
                    *self = vec![value].into();
                } else {
                    panic!("invalid ThinVec::insert");
                }
            }
            ThinVec(Some(ref mut vec)) => vec.insert(index, value),
        }
    }

    pub fn remove(&mut self, index: usize) -> T {
        match self {
            ThinVec(None) => panic!("invalid ThinVec::remove"),
            ThinVec(Some(vec)) => vec.remove(index),
        }
    }

    pub fn as_slice(&self) -> &[T] {
        match self {
            ThinVec(None) => &[],
            ThinVec(Some(vec)) => vec.as_slice(),
        }
    }
}

impl<T> From<Vec<T>> for ThinVec<T> {
    fn from(vec: Vec<T>) -> Self {
        if vec.is_empty() { ThinVec(None) } else { ThinVec(Some(Box::new(vec))) }
    }
}

impl<T> Into<Vec<T>> for ThinVec<T> {
    fn into(self) -> Vec<T> {
        match self {
            ThinVec(None) => Vec::new(),
            ThinVec(Some(vec)) => *vec,
        }
    }
}

impl<T> ::std::ops::Deref for ThinVec<T> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        match *self {
            ThinVec(None) => &[],
            ThinVec(Some(ref vec)) => vec,
        }
    }
}

impl<T> ::std::ops::DerefMut for ThinVec<T> {
    fn deref_mut(&mut self) -> &mut [T] {
        match *self {
            ThinVec(None) => &mut [],
            ThinVec(Some(ref mut vec)) => vec,
        }
    }
}

impl<T> FromIterator<T> for ThinVec<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        // `Vec::from_iter()` should not allocate if the iterator is empty.
        let vec: Vec<_> = iter.into_iter().collect();
        if vec.is_empty() { ThinVec(None) } else { ThinVec(Some(Box::new(vec))) }
    }
}

impl<T> IntoIterator for ThinVec<T> {
    type Item = T;
    type IntoIter = std::vec::IntoIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        // This is still performant because `Vec::new()` does not allocate.
        self.0.map_or_else(Vec::new, |ptr| *ptr).into_iter()
    }
}

impl<'a, T> IntoIterator for &'a ThinVec<T> {
    type Item = &'a T;
    type IntoIter = std::slice::Iter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.as_ref().iter()
    }
}

impl<'a, T> IntoIterator for &'a mut ThinVec<T> {
    type Item = &'a mut T;
    type IntoIter = std::slice::IterMut<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.as_mut().iter_mut()
    }
}

impl<T> Extend<T> for ThinVec<T> {
    fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        match *self {
            ThinVec(Some(ref mut vec)) => vec.extend(iter),
            ThinVec(None) => *self = iter.into_iter().collect::<Vec<_>>().into(),
        }
    }

    fn extend_one(&mut self, item: T) {
        self.push(item)
    }

    fn extend_reserve(&mut self, additional: usize) {
        match *self {
            ThinVec(Some(ref mut vec)) => vec.reserve(additional),
            ThinVec(None) => *self = Vec::with_capacity(additional).into(),
        }
    }
}

impl<T: HashStable<CTX>, CTX> HashStable<CTX> for ThinVec<T> {
    fn hash_stable(&self, hcx: &mut CTX, hasher: &mut StableHasher) {
        (**self).hash_stable(hcx, hasher)
    }
}

impl<T> Default for ThinVec<T> {
    fn default() -> Self {
        Self(None)
    }
}

#[cfg(test)]
mod tests;
