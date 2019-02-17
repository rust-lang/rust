use std::{hash, ptr};
use std::ops::Deref;

/// A wrapper around reference that compares and hashes like a pointer.
/// Can be used as a key in sets/maps indexed by pointers to avoid `unsafe`.
#[derive(Debug)]
pub struct PtrKey<'a, T>(pub &'a T);

impl<'a, T> Clone for PtrKey<'a, T> {
    fn clone(&self) -> Self { *self }
}

impl<'a, T> Copy for PtrKey<'a, T> {}

impl<'a, T> PartialEq for PtrKey<'a, T> {
    fn eq(&self, rhs: &Self) -> bool {
        ptr::eq(self.0, rhs.0)
    }
}

impl<'a, T> Eq for PtrKey<'a, T> {}

impl<'a, T> hash::Hash for PtrKey<'a, T> {
    fn hash<H: hash::Hasher>(&self, hasher: &mut H) {
        (self.0 as *const T).hash(hasher)
    }
}

impl<'a, T> Deref for PtrKey<'a, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        self.0
    }
}
