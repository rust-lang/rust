//! A doubly-linked list where callers are in charge of memory allocation
//! of the nodes in the list.

#[cfg(test)]
mod tests;

use crate::ptr::NonNull;

pub struct UnsafeListEntry<T> {
    next: Option<NonNull<UnsafeListEntry<T>>>,
    prev: Option<NonNull<UnsafeListEntry<T>>>,
    value: T,
}

impl<T> UnsafeListEntry<T> {
    pub fn new(value: T) -> Self {
        UnsafeListEntry { next: None, prev: None, value }
    }
}

pub struct UnsafeList<T> {
    head_tail: Option<(NonNull<UnsafeListEntry<T>>, NonNull<UnsafeListEntry<T>>)>,
}

impl<T> UnsafeList<T> {
    pub const fn new() -> Self {
        UnsafeList { head_tail: None }
    }

    /// Pushes an entry onto the back of the list.
    ///
    /// # Safety
    ///
    /// `entry` must be valid for writes to an `UnsafeListEntry` that may not be
    /// currently part of an `UnsafeList`, and must remain valid until the entry
    /// has been removed from the list.
    ///
    /// The immutable reference to the value returned by this function is valid
    /// for as long as `entry` remains valid. Notably, it will not be invalidated
    /// by operations on `self`.
    pub unsafe fn push<'a>(&mut self, entry: NonNull<UnsafeListEntry<T>>) -> &'a T {
        if let Some((head, tail)) = self.head_tail {
            // SAFETY: `tail` belongs to the current list and therefore its `next`
            //         field must be writable.
            unsafe { (*tail.as_ptr()).next = Some(entry) };
            self.head_tail = Some((head, entry));
        } else {
            // The list was previously empty, so the new entry is both the head
            // and tail node.
            self.head_tail = Some((entry, entry));
        }

        // SAFETY: the value field is only accessed via shared reference for as
        //         long as the entry is part of the list.
        unsafe { &(*entry.as_ptr()).value }
    }

    /// Pops an entry from the front of the list.
    pub fn pop(&mut self) -> Option<&T> {
        let (head, tail) = self.head_tail?;
        if let Some(next) = unsafe { (*head.as_ptr()).next } {
            // SAFETY: the `next` node must still be part of the list, and thus
            //         its `prev` pointer must be writable.
            unsafe { (*next.as_ptr()).prev = None };
            self.head_tail = Some((next, tail));
        } else {
            // There is only a single node in the list, so from now on it will
            // be empty.
            self.head_tail = None;
        }

        // SAFETY: the entry pointer passed to `push` may only be invalidated
        // once the removal from the list has been observed, which requires
        // either access to the list (which would also mark the end of the
        // lifetime of this references since we have a mutable reference) or
        // another mechanism, where the caller of this function communicates
        // the removal to the thread in question and thus is aware of the
        // potential invalidation of this reference.
        Some(unsafe { &(*head.as_ptr()).value })
    }

    /// Removes a given entry from the list.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `entry` has been pushed onto `self`
    /// prior to this call and has not been removed since then. This implies
    /// that `entry` must have the same provenance as the pointer passed to
    /// `push`.
    pub unsafe fn remove(&mut self, entry: NonNull<UnsafeListEntry<T>>) {
        let Some((head, tail)) = self.head_tail else {
            rtabort!("list cannot be empty");
        };

        // SAFETY: `entry` must be in the list, so its `prev` field must be
        // accessible.
        let prev = unsafe { (*entry.as_ptr()).prev };
        // SAFETY: same argument as above.
        let next = unsafe { (*entry.as_ptr()).next };

        match (prev, next) {
            // SAFETY: same argument as above, these nodes are in the list.
            (Some(prev), Some(next)) => unsafe {
                (*prev.as_ptr()).next = Some(next);
                (*next.as_ptr()).prev = Some(prev);
            },
            // SAFETY: same argument as above, these nodes are in the list.
            (Some(prev), None) => unsafe {
                (*prev.as_ptr()).next = None;
                self.head_tail = Some((head, prev));
            },
            // SAFETY: same argument as above, these nodes are in the list.
            (None, Some(next)) => unsafe {
                (*next.as_ptr()).prev = None;
                self.head_tail = Some((next, tail));
            },
            (None, None) => {
                // There is only a single node in the list, so from now on it
                // will be empty.
                self.head_tail = None;
            }
        }
    }
}
