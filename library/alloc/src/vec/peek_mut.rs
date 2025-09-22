use core::ops::{Deref, DerefMut};

use super::Vec;
use crate::alloc::{Allocator, Global};
use crate::fmt;

/// Structure wrapping a mutable reference to the last item in a
/// `Vec`.
///
/// This `struct` is created by the [`peek_mut`] method on [`Vec`]. See
/// its documentation for more.
///
/// [`peek_mut`]: Vec::peek_mut
#[unstable(feature = "vec_peek_mut", issue = "122742")]
pub struct PeekMut<
    'a,
    T,
    #[unstable(feature = "allocator_api", issue = "32838")] A: Allocator = Global,
> {
    vec: &'a mut Vec<T, A>,
}

#[unstable(feature = "vec_peek_mut", issue = "122742")]
impl<T: fmt::Debug, A: Allocator> fmt::Debug for PeekMut<'_, T, A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("PeekMut").field(self.deref()).finish()
    }
}

impl<'a, T, A: Allocator> PeekMut<'a, T, A> {
    pub(super) fn new(vec: &'a mut Vec<T, A>) -> Option<Self> {
        if vec.is_empty() { None } else { Some(Self { vec }) }
    }

    /// Removes the peeked value from the vector and returns it.
    #[unstable(feature = "vec_peek_mut", issue = "122742")]
    pub fn pop(this: Self) -> T {
        // SAFETY: PeekMut is only constructed if the vec is non-empty
        unsafe { this.vec.pop().unwrap_unchecked() }
    }
}

#[unstable(feature = "vec_peek_mut", issue = "122742")]
impl<'a, T, A: Allocator> Deref for PeekMut<'a, T, A> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        let idx = self.vec.len() - 1;
        // SAFETY: PeekMut is only constructed if the vec is non-empty
        unsafe { self.vec.get_unchecked(idx) }
    }
}

#[unstable(feature = "vec_peek_mut", issue = "122742")]
impl<'a, T, A: Allocator> DerefMut for PeekMut<'a, T, A> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        let idx = self.vec.len() - 1;
        // SAFETY: PeekMut is only constructed if the vec is non-empty
        unsafe { self.vec.get_unchecked_mut(idx) }
    }
}
