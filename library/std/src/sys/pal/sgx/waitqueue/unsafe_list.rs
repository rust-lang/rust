//! A doubly-linked list where callers are in charge of memory allocation
//! of the nodes in the list.
//!
//! # Safety
//!
//! `UnsafeList` itself does not synchronize any of its memory accesses, so
//! callers must serialize all operations on a list, e.g. with a lock.
//!
//! While an entry passed to `push` is in the list, it must not be invalidated,
//! with one exception explained below. Invalidation of the entry, by creating a new
//! exclusive reference to it, would invalidate the pointers to the entry stored
//! in the list. The entry goes through one of two flows (see also each operation's
//! safety documentation):
//!
//! * `push` -> `pop`, usually with `pop` on another thread: the entry pointer
//!   stored in the list keeps its `push`-time provenance. As mentioned, for it to
//!   still be valid to dereference in `pop`, the pushing caller must not access
//!   the entry in between. After `pop`, the references into `value` returned by
//!   `push` and `pop` are held concurrently, possibly by two threads. This is
//!   valid as they are shared references, but mutating `value` requires interior
//!   mutability and synchronization. That synchronization must also ensure the
//!   entry is only deallocated after the popping thread's last access to it.
//! * `push` -> `remove`, on the thread that pushed: the caller reclaims a pushed
//!   entry by passing a reference to the entry to `remove`. The entry must still
//!   be in the list. The caller of `remove` must create a new exclusive reference
//!   to the entry, which invalidates the pointers to the entry stored in the list.
//!   This is fine in this case because `remove` only overwrites those pointers,
//!   and never dereferences them.

// # Aliasing
//
// The list is self-referential: it stores pointers to its own `head_tail`
// field in the list entries' links. `UnsafePinned` is used to ensure pointer
// validity.
//
// Pointers to the other entries are derived from the exclusive reference passed
// to `push` and stay valid while the entry is in the list (see the safety
// requirements in the module documentation). Multiple immutable references may
// exist to values of entries in the list, while the links in the list may be
// mutated simultaneously. Creating mutable references to entries to update the
// links would invalidate any outstanding shared references. As such, all links
// are updated via raw-pointer place expressions instead, keeping the value
// references valid.
//
// # Pointer dereferencing
//
// All pointers stored in the list are valid to dereference:
//
// 1. The head/tail pointer is derived from `head_tail`'s `UnsafePinned`
//    wherever it is needed. Because of the `UnsafePinned` wrapper, no
//    exclusive reference to the list (or a structure containing it) makes an
//    aliasing claim on `head_tail`, so every derived pointer and every copy
//    of it stored in the links stay valid for the list's lifetime.
// 2. Pointers to other entries, stored in the links, are derived from the
//    exclusive reference passed to `push` and stay valid while the entry is
//    in the list, as ensured by the safety requirements in the module
//    documentation.
//
// Both points rely on this code never creating references to entries, as
// those would make their own aliasing claims on the entries.

#[cfg(test)]
mod tests;

use crate::pin::UnsafePinned;
use crate::ptr::{self, NonNull};

/// A caller-allocated list entry.
///
/// While the entry is in a list, the list holds a pointer derived from the
/// exclusive reference passed to `UnsafeList::push`, so the caller must not
/// access the entry until it is removed from the list. `UnsafeList::push`
/// returns a reference borrowing the entry, and `UnsafeList::remove`
/// reborrows it exclusively, so the borrow checker enforces this for safe
/// accesses.
pub struct UnsafeListEntry<T> {
    next: NonNull<UnsafeListEntry<T>>,
    prev: NonNull<UnsafeListEntry<T>>,
    value: Option<T>,
}

impl<T> UnsafeListEntry<T> {
    const fn dummy() -> Self {
        UnsafeListEntry { next: NonNull::dangling(), prev: NonNull::dangling(), value: None }
    }

    pub fn new(value: T) -> Self {
        UnsafeListEntry { value: Some(value), ..Self::dummy() }
    }
}

// WARNING: self-referential struct!
pub struct UnsafeList<T> {
    // UnsafePinned isn't required to implement this code, but it makes it a lot
    // simpler. Without UnsafePinned, the provenance of each entry link pointer
    // would need to be re-established prior to dereferencing, whenever it points
    // to `head_tail`.
    head_tail: UnsafePinned<UnsafeListEntry<T>>,
    init: bool,
}

impl<T> UnsafeList<T> {
    pub const fn new() -> Self {
        UnsafeList { head_tail: UnsafePinned::new(UnsafeListEntry::dummy()), init: false }
    }

    fn head_tail(&mut self) -> NonNull<UnsafeListEntry<T>> {
        // SAFETY: `get_mut_unchecked` returns the address of `head_tail`,
        // which is non-null.
        unsafe { NonNull::new_unchecked(self.head_tail.get_mut_unchecked()) }
    }

    /// # Safety
    ///
    /// The caller must ensure the list is never moved after this call: the
    /// list becomes self-referential.
    unsafe fn init(&mut self) {
        if !self.init {
            let head_tail = self.head_tail();
            // SAFETY: `head_tail` is valid to dereference (see point 1 of the
            // `Pointer dereferencing` explanation at the top of the file).
            unsafe { (*head_tail.as_ptr()).next = head_tail };
            unsafe { (*head_tail.as_ptr()).prev = head_tail };
            self.init = true;
        }
    }

    pub fn is_empty(&self) -> bool {
        if self.init {
            // SAFETY: `get` returns the address of `head_tail`, which is
            // non-null.
            let head_tail = unsafe { NonNull::new_unchecked(self.head_tail.get()) };
            // SAFETY: `head_tail` is valid to dereference (see point 1
            // of the `Pointer dereferencing` explanation at the top of the
            // file).
            let first = unsafe { (*head_tail.as_ptr()).next };
            if first == head_tail {
                // ,-------> /---------\ next ---,
                // |         |head_tail|         |
                // `--- prev \---------/ <-------`
                // SAFETY: `head_tail` is valid to dereference.
                unsafe { rtassert!((*head_tail.as_ptr()).prev == first) };
                true
            } else {
                false
            }
        } else {
            true
        }
    }

    /// Pushes an entry onto the back of the list.
    ///
    /// # Safety
    ///
    /// The entry must remain allocated until the entry is removed from the
    /// list AND the caller who popped is done using the entry. Special
    /// care must be taken in the caller of `push` to ensure unwinding does
    /// not destroy the stack frame containing the entry. While the entry is
    /// in the list, it must not be accessed except through the reference
    /// returned here or by passing the entry to `remove`.
    pub unsafe fn push<'a>(&mut self, entry: &'a mut UnsafeListEntry<T>) -> &'a T {
        unsafe { self.init() };

        // BEFORE:
        //     /---------\ next ---> /---------\
        // ... |prev_tail|           |head_tail| ...
        //     \---------/ <--- prev \---------/
        //
        // AFTER:
        //     /---------\ next ---> /-----\ next ---> /---------\
        // ... |prev_tail|           |entry|           |head_tail| ...
        //     \---------/ <--- prev \-----/ <--- prev \---------/
        let entry = unsafe { NonNull::new_unchecked(entry) };
        let head_tail = self.head_tail();
        // SAFETY: `head_tail` is valid to dereference (see point 1
        // of the `Pointer dereferencing` explanation at the top of the
        // file).
        let prev_tail = unsafe { ptr::replace(&raw mut (*head_tail.as_ptr()).prev, entry) };
        // SAFETY: `entry` is valid to dereference: it was derived from an
        // exclusive reference above.
        unsafe { (*entry.as_ptr()).prev = prev_tail };
        unsafe { (*entry.as_ptr()).next = head_tail };
        // SAFETY: `prev_tail` was loaded from the list's links, so it is
        // valid to dereference (see points 1 and 2 of the
        // `Pointer dereferencing` explanation at the top of the file).
        unsafe { (*prev_tail.as_ptr()).next = entry };
        // unwrap ok: always `Some` on non-dummy entries
        unsafe { (*entry.as_ptr()).value.as_ref() }.unwrap()
    }

    /// Pops an entry from the front of the list.
    ///
    /// # Safety
    ///
    /// The caller must make sure to synchronize ending the borrow of the
    /// return value and deallocation of the containing entry.
    pub unsafe fn pop<'a>(&mut self) -> Option<&'a T> {
        if self.is_empty() {
            None
        } else {
            // BEFORE:
            //     /---------\ next ---> /-----\ next ---> /------\
            // ... |head_tail|           |first|           |second| ...
            //     \---------/ <--- prev \-----/ <--- prev \------/
            //
            // AFTER:
            //     /---------\ next ---> /------\
            // ... |head_tail|           |second| ...
            //     \---------/ <--- prev \------/

            let head_tail = self.head_tail();
            // SAFETY: `head_tail` is valid to dereference (see point 1
            // of the `Pointer dereferencing` explanation at the top of the
            // file).
            let first = unsafe { (*head_tail.as_ptr()).next };
            // SAFETY: `first` was loaded from the list's links, so it is
            // valid to dereference (see point 2 of the
            // `Pointer dereferencing` explanation at the top of the file).
            let second = unsafe { (*first.as_ptr()).next };
            unsafe { (*head_tail.as_ptr()).next = second };
            // SAFETY: `second` was loaded from the list's links, so it is
            // valid to dereference (see points 1 and 2 of the
            // `Pointer dereferencing` explanation at the top of the file).
            unsafe { (*second.as_ptr()).prev = head_tail };
            unsafe { (*first.as_ptr()).next = NonNull::dangling() };
            unsafe { (*first.as_ptr()).prev = NonNull::dangling() };
            // unwrap ok: always `Some` on non-dummy entries
            Some(unsafe { (*first.as_ptr()).value.as_ref() }.unwrap())
        }
    }

    /// Removes an entry from the list.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `entry` has been pushed onto `self`
    /// prior to this call, has not been removed from the list since then
    /// (by `pop` or `remove`), and has not moved since it was pushed.
    pub unsafe fn remove(&mut self, entry: &mut UnsafeListEntry<T>) {
        rtassert!(!self.is_empty());
        // BEFORE:
        //     /----\ next ---> /-----\ next ---> /----\
        // ... |prev|           |entry|           |next| ...
        //     \----/ <--- prev \-----/ <--- prev \----/
        //
        // AFTER:
        //     /----\ next ---> /----\
        // ... |prev|           |next| ...
        //     \----/ <--- prev \----/

        // The exclusive reference `entry`, created by the caller, has
        // invalidated the pointers to `entry` stored in its neighbors (see
        // the module documentation); those are only overwritten below,
        // never dereferenced.
        let prev = entry.prev;
        let next = entry.next;
        // SAFETY: `prev` and `next` were loaded from `entry`'s links, so
        // they are valid to dereference (see points 1 and 2 of the
        // `Pointer dereferencing` explanation at the top of the file).
        unsafe { (*prev.as_ptr()).next = next };
        unsafe { (*next.as_ptr()).prev = prev };
        entry.next = NonNull::dangling();
        entry.prev = NonNull::dangling();
    }
}
