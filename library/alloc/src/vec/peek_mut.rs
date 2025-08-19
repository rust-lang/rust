use core::ops::{Deref, DerefMut};

use super::Vec;
use crate::fmt;

/// Structure wrapping a mutable reference to the last item in a
/// `Vec`.
///
/// This `struct` is created by the [`peek_mut`] method on [`Vec`]. See
/// its documentation for more.
///
/// [`peek_mut`]: Vec::peek_mut
#[unstable(feature = "vec_peek_mut", issue = "122742")]
pub struct PeekMut<'a, T> {
    vec: &'a mut Vec<T>,
}

#[unstable(feature = "vec_peek_mut", issue = "122742")]
impl<T: fmt::Debug> fmt::Debug for PeekMut<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("PeekMut").field(self.deref()).finish()
    }
}

impl<'a, T> PeekMut<'a, T> {
    pub(crate) fn new(vec: &'a mut Vec<T>) -> Option<Self> {
        if vec.is_empty() { None } else { Some(Self { vec }) }
    }

    /// Removes the peeked value from the vector and returns it.
    #[unstable(feature = "vec_peek_mut", issue = "122742")]
    pub fn pop(self) -> T {
        // SAFETY: PeekMut is only constructed if the vec is non-empty
        unsafe { self.vec.pop().unwrap_unchecked() }
    }
}

#[unstable(feature = "vec_peek_mut", issue = "122742")]
impl<'a, T> Deref for PeekMut<'a, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        // SAFETY: PeekMut is only constructed if the vec is non-empty
        unsafe { self.vec.get_unchecked(self.vec.len() - 1) }
    }
}

#[unstable(feature = "vec_peek_mut", issue = "122742")]
impl<'a, T> DerefMut for PeekMut<'a, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        let idx = self.vec.len() - 1;
        // SAFETY: PeekMut is only constructed if the vec is non-empty
        unsafe { self.vec.get_unchecked_mut(idx) }
    }
}
