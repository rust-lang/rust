use crate::stable_hasher::{HashStable, StableHasher};

/// A vector type optimized for cases where this size is usually 0 (cf. `SmallVector`).
/// The `Option<Box<..>>` wrapping allows us to represent a zero sized vector with `None`,
/// which uses only a single (null) pointer.
#[derive(Clone, RustcEncodable, RustcDecodable, Debug)]
pub struct ThinVec<T>(Option<Box<Vec<T>>>);

impl<T> ThinVec<T> {
    pub fn new() -> Self {
        ThinVec(None)
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
