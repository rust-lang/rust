use crate::stable_hasher::{StableHasher, StableHasherResult, HashStable};

/// A vector type optimized for cases where this size is usually 0 (cf. `SmallVector`).
/// The `Option<Box<..>>` wrapping allows us to represent a zero sized vector with `None`,
/// which uses only a single (null) pointer.
#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug)]
pub struct ThinVec<T>(Option<Box<Vec<T>>>);

impl<T> ThinVec<T> {
    pub fn new() -> Self {
        ThinVec(None)
    }
}

impl<T> From<Vec<T>> for ThinVec<T> {
    fn from(vec: Vec<T>) -> Self {
        if vec.is_empty() {
            ThinVec(None)
        } else {
            ThinVec(Some(Box::new(vec)))
        }
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
}

impl<T: HashStable<CTX>, CTX> HashStable<CTX> for ThinVec<T> {
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut CTX,
                                          hasher: &mut StableHasher<W>) {
        (**self).hash_stable(hcx, hasher)
    }
}

impl<T> Default for ThinVec<T> {
    fn default() -> Self {
        Self(None)
    }
}
