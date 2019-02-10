use crate::ops::{Deref, DerefMut};

#[derive(Copy, Clone, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(align(64))]
pub(super) struct Aligner;

#[derive(Copy, Clone, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(super) struct CacheAligned<T>(pub T, pub Aligner);

impl<T> Deref for CacheAligned<T> {
     type Target = T;
     fn deref(&self) -> &Self::Target {
         &self.0
     }
}

impl<T> DerefMut for CacheAligned<T> {
     fn deref_mut(&mut self) -> &mut Self::Target {
         &mut self.0
     }
}

impl<T> CacheAligned<T> {
    pub(super) fn new(t: T) -> Self {
        CacheAligned(t, Aligner)
    }
}
