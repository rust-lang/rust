use crate::stable_hasher::{HashStable, StableHasher};

use std::iter::FromIterator;

/// A vector type optimized for cases where this size is usually 0 (cf. `SmallVec`).
/// The `Option<Box<..>>` wrapping allows us to represent a zero sized vector with `None`,
/// which uses only a single (null) pointer.
#[derive(Clone, Encodable, Decodable, Debug)]
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
        match *self {
            ThinVec(Some(ref mut vec)) => vec.push(item),
            ThinVec(None) => *self = vec![item].into(),
        }
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
