//! A doubly-linked list where callers are in charge of memory allocation
//! of the nodes in the list.
//!
//! # Aliasing
//!
//! To avoid aliasing violations, the implementation follows these rules:
//!
//! * References are converted to raw pointers as soon as they are taken as
//!   input (`UnsafeListEntry::entry`), and pointers are converted back to
//!   references as late as possible, just before returning an output.
//! * Pointers are stored, loaded and compared as `EntryPtr`, which cannot be
//!   dereferenced. All memory accesses go through a reference or through
//!   `DerefPtr` (obtained from `UnsafeListEntry::entry` or
//!   `UnsafeList::launder`), whose raw-pointer place expressions
//!   (`(*ptr.as_ptr()).field`) access memory with the pointer's own
//!   provenance: no intermediate reference with its own claim on the memory
//!   is created, so no other pointer is invalidated.
//! * `head_tail_entry` lives inside the list itself, so a head/tail pointer
//!   stored across method calls would be invalidated whenever a new
//!   exclusive reference to the list (or a structure containing it) is
//!   created. Therefore only the head/tail *address* is stored
//!   (`UnsafeList::head_tail`); an exclusive reference with `&mut self`'s
//!   provenance is reconstructed at the point of use (`fn head_tail()`), whose
//!   borrow lets the compiler enforce that only the most recent derivation is
//!   used. Pointers loaded from the list's links are laundered before every
//!   dereference (`fn launder()`), refreshing a stale head/tail pointer. The
//!   resulting `DerefPtr` borrows the list so that the borrow checker enforces
//!   it can only be used before the next access to the list.

#[cfg(test)]
mod tests;

use self::ptr::{DerefPtr, EntryPtr};
use crate::cell::Cell;
use crate::num::NonZero;
use crate::ptr::NonNull;

struct UnsafeListEntryInner<T> {
    next: EntryPtr<T>,
    prev: EntryPtr<T>,
    value: Option<T>,
}

impl<T> UnsafeListEntryInner<T> {
    fn dummy() -> Self {
        UnsafeListEntryInner { next: EntryPtr::dangling(), prev: EntryPtr::dangling(), value: None }
    }

    fn new(value: T) -> Self {
        UnsafeListEntryInner { value: Some(value), ..Self::dummy() }
    }
}

/// A caller-allocated list entry.
///
/// While the entry is in a list, the list holds a pointer derived from the
/// exclusive reference passed to `UnsafeList::push`, so the caller must not
/// access the entry until it is removed from the list. `UnsafeList::push`
/// returns a reference borrowing the entry, and `UnsafeList::remove` reborrow
/// it exclusively, so the borrow checker enforces this for safe accesses.
pub struct UnsafeListEntry<T> {
    // Interior mutability: while the entry is in a list, the list mutates the
    // links even though the caller still owns the entry.
    inner: Cell<UnsafeListEntryInner<T>>,
}

impl<T> UnsafeListEntry<T> {
    pub fn new(value: T) -> Self {
        UnsafeListEntry { inner: Cell::new(UnsafeListEntryInner::new(value)) }
    }

    fn inner(&mut self) -> DerefPtr<'_, T> {
        // SAFETY: the pointer is derived from the exclusive reference to the
        // entry's storage, which the returned `DerefPtr` borrows for its
        // lifetime, so it is valid for dereferences while the result lives.
        unsafe { DerefPtr::new(self.inner.get_mut().into()) }
    }
}

// The pointer types of the list. They are in a module to enforce the
// privacy of their fields: outside code can only construct and convert
// them through their methods, which uphold the invariants described here.
mod ptr {
    use super::UnsafeListEntryInner;
    use crate::marker::PhantomData;
    use crate::num::NonZero;
    use crate::ptr::NonNull;

    /// A pointer to a list entry that cannot be dereferenced.
    ///
    /// This is the only form in which the list stores, loads and compares
    /// pointers. A stored head/tail pointer's provenance may be stale (see
    /// `UnsafeList::head_tail`), so an `EntryPtr` is always valid to store
    /// and compare, but never to dereference. Dereferencing requires a
    /// reference or a `DerefPtr`.
    pub(super) struct EntryPtr<T>(NonNull<UnsafeListEntryInner<T>>);

    // Not derived to avoid the implied `T:` bounds.
    impl<T> Clone for EntryPtr<T> {
        fn clone(&self) -> Self {
            *self
        }
    }

    impl<T> Copy for EntryPtr<T> {}

    impl<T> PartialEq for EntryPtr<T> {
        fn eq(&self, other: &Self) -> bool {
            self.0 == other.0
        }
    }

    impl<T> From<&mut UnsafeListEntryInner<T>> for EntryPtr<T> {
        fn from(entry: &mut UnsafeListEntryInner<T>) -> Self {
            EntryPtr(NonNull::from(entry))
        }
    }

    impl<T> EntryPtr<T> {
        pub(super) fn dangling() -> Self {
            EntryPtr(NonNull::dangling())
        }

        pub(super) fn addr(self) -> NonZero<usize> {
            self.0.addr()
        }
    }

    /// A dereferenceable pointer to a list entry.
    ///
    /// This is the only pointer type in this module that can be dereferenced
    /// (`as_ptr`). The `PhantomData` exclusively borrows the structure the
    /// pointee's storage belongs to: the `UnsafeListEntry` for a pointer
    /// obtained from `UnsafeListEntry::entry`, or the list for a pointer
    /// obtained from `UnsafeList::launder`. In the latter case, while this
    /// pointer is live the borrow checker rejects any other access to the
    /// list, which would invalidate this pointer if it is the head/tail
    /// pointer (see `UnsafeList::head_tail`); this enforces that the pointer
    /// can only be used before the next access to the list. Converting it
    /// into an `EntryPtr` (`into_ptr`) ends that protection; the value may
    /// then only be stored and compared, not dereferenced.
    ///
    /// Unlike a reference, this type makes no aliasing claim on the pointee:
    /// when it points to an entry, another thread may concurrently hold the
    /// reference into the entry's `value` field. As such, using exclusive
    /// references to the whole entry would be invalid.
    pub(super) struct DerefPtr<'a, T> {
        ptr: NonNull<UnsafeListEntryInner<T>>,
        borrow: PhantomData<&'a mut UnsafeListEntryInner<T>>,
    }

    impl<'a, T> DerefPtr<'a, T> {
        /// Creates a dereferenceable pointer from `ptr`.
        ///
        /// # Safety
        ///
        /// The caller must ensure that `ptr`'s provenance is valid for
        /// accessing the pointee wherever the result is dereferenced, and
        /// pick `'a` such that the borrow checker upholds this (see
        /// `UnsafeListEntry::entry` and `UnsafeList::launder`).
        pub(super) unsafe fn new(ptr: EntryPtr<T>) -> Self {
            DerefPtr { ptr: ptr.0, borrow: PhantomData }
        }

        pub(super) fn as_ptr(&self) -> *mut UnsafeListEntryInner<T> {
            self.ptr.as_ptr()
        }

        pub(super) fn into_ptr(self) -> EntryPtr<T> {
            EntryPtr(self.ptr)
        }
    }
}

// WARNING: self-referential struct!
pub struct UnsafeList<T> {
    // The head/tail *address*. See the module documentation for why this can't
    // be a pointer.
    head_tail: NonZero<usize>,
    head_tail_entry: Option<UnsafeListEntryInner<T>>,
}

impl<T> UnsafeList<T> {
    pub const fn new() -> Self {
        unsafe { UnsafeList { head_tail: NonZero::new_unchecked(1), head_tail_entry: None } }
    }

    fn init(&mut self) {
        if self.head_tail_entry.is_none() {
            self.head_tail_entry = Some(UnsafeListEntryInner::dummy());
            // unwrap ok: `head_tail_entry` was assigned `Some` above
            let head_tail = self.head_tail_entry.as_mut().unwrap();
            // `ptr` is stored as a value only; loads are laundered before
            // being dereferenced.
            let ptr = EntryPtr::from(&mut *head_tail);
            head_tail.next = ptr;
            head_tail.prev = ptr;
            self.head_tail = ptr.addr();
        }
    }

    /// Returns a reference to `head_tail_entry`, which must have been
    /// initialized by `init`.
    ///
    /// The reference is derived from `&mut self` — whose provenance spans the
    /// entire list, including `head_tail_entry` — and borrows the list: while
    /// it is live, the borrow checker rejects any other access to the list,
    /// including another `head_tail` or `launder` call. Converting it into a
    /// pointer (`EntryPtr::from`) ends that protection: such a pointer is
    /// invalidated by the next exclusive reference to the list (see the
    /// module docs) and may then only be used as a value. Head/tail pointers
    /// loaded from the list's links carry stale provenance and must be
    /// passed through `launder` before being dereferenced.
    fn head_tail(&mut self) -> &mut UnsafeListEntryInner<T> {
        let head_tail = self.head_tail;
        let mut ptr = NonNull::from(self).cast::<UnsafeListEntryInner<T>>().with_addr(head_tail);
        // SAFETY: `ptr` was derived from `&mut self` just above and points to
        // the initialized `head_tail_entry`. No other reference to it can
        // exist: references into entries never point at it, and the borrow
        // of `self` excludes all other access to the list.
        unsafe { ptr.as_mut() }
    }

    /// Returns a dereferenceable version of `ptr`, a pointer loaded from the
    /// list's links.
    ///
    /// If `ptr` is a head/tail pointer, it carries stale provenance and is
    /// replaced by a fresh derivation with the same address.
    ///
    /// # Safety
    ///
    /// `ptr` must be `self`'s head/tail pointer or point to an entry that is
    /// currently in `self`. That should be the case for any pointer that is
    /// currently stored in the list.
    unsafe fn launder(&mut self, ptr: EntryPtr<T>) -> DerefPtr<'_, T> {
        let ptr = if ptr.addr() == self.head_tail { EntryPtr::from(self.head_tail()) } else { ptr };
        // SAFETY: per this function's contract, `ptr` is now either the
        // freshly derived head/tail pointer — valid until the next access to
        // the list, which the result's borrow of `self` prevents while it
        // lives — or a pointer to an entry in the list, whose provenance
        // stems from the exclusive reference passed to `push` and remains
        // valid while the entry is in the list.
        unsafe { DerefPtr::new(ptr) }
    }

    pub fn is_empty(&mut self) -> bool {
        if self.head_tail_entry.is_some() {
            let head_tail = self.head_tail();
            let first = head_tail.next;
            let prev = head_tail.prev;
            if first == EntryPtr::from(head_tail) {
                // ,-------> /---------\ next ---,
                // |         |head_tail|         |
                // `--- prev \---------/ <-------`
                rtassert!(prev == first);
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
        self.init();

        // BEFORE:
        //     /---------\ next ---> /---------\
        // ... |prev_tail|           |head_tail| ...
        //     \---------/ <--- prev \---------/
        //
        // AFTER:
        //     /---------\ next ---> /-----\ next ---> /---------\
        // ... |prev_tail|           |entry|           |head_tail| ...
        //     \---------/ <--- prev \-----/ <--- prev \---------/
        let entry = entry.inner();
        let head_tail = self.head_tail();
        let prev_tail = head_tail.prev;
        // SAFETY: `entry` was derived from an exclusive reference above.
        unsafe { (*entry.as_ptr()).next = EntryPtr::from(&mut *head_tail) };
        unsafe { (*entry.as_ptr()).prev = prev_tail };
        // unwrap ok: always `Some` on non-dummy entries
        let ret = unsafe { (*entry.as_ptr()).value.as_ref() }.unwrap();
        let entry = entry.into_ptr();
        head_tail.prev = entry;
        // SAFETY: `prev_tail` was loaded from the head/tail entry's links
        let prev_tail = unsafe { self.launder(prev_tail) };
        // SAFETY: `prev_tail` is dereferenceable (see `DerefPtr`).
        unsafe { (*prev_tail.as_ptr()).next = entry };
        ret
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
            let head_tail = self.head_tail();

            // BEFORE:
            //     /---------\ next ---> /-----\ next ---> /------\
            // ... |head_tail|           |first|           |second| ...
            //     \---------/ <--- prev \-----/ <--- prev \------/
            //
            // AFTER:
            //     /---------\ next ---> /------\
            // ... |head_tail|           |second| ...
            //     \---------/ <--- prev \------/
            let first = head_tail.next;
            let head_tail = EntryPtr::from(head_tail);
            // SAFETY: `first` was loaded from the head/tail entry's links
            let first = unsafe { self.launder(first) };
            // SAFETY: `first` is dereferenceable (see `DerefPtr`).
            let second = unsafe { (*first.as_ptr()).next };
            unsafe { (*first.as_ptr()).next = EntryPtr::dangling() };
            unsafe { (*first.as_ptr()).prev = EntryPtr::dangling() };
            // unwrap ok: always `Some` on non-dummy entries
            let ret = unsafe { (*first.as_ptr()).value.as_ref() }.unwrap();
            self.head_tail().next = second;
            // SAFETY: `second` was loaded from `first`'s links while `first`
            // was in the list
            let second = unsafe { self.launder(second) };
            // SAFETY: `second` is dereferenceable (see `DerefPtr`).
            unsafe { (*second.as_ptr()).prev = head_tail };
            Some(ret)
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

        // Deriving a fresh pointer to `entry` invalidates the pointers to
        // `entry` stored in its neighbors, but those are only overwritten
        // below, never dereferenced.
        let entry = entry.inner();
        // SAFETY: `entry` is dereferenceable (see `DerefPtr`).
        let prev = unsafe { (*entry.as_ptr()).prev };
        let next = unsafe { (*entry.as_ptr()).next };
        // SAFETY: `prev` was loaded from `entry`'s links
        let prev = unsafe { self.launder(prev) };
        // SAFETY: `prev` is dereferenceable (see `DerefPtr`).
        unsafe { (*prev.as_ptr()).next = next };
        let prev = prev.into_ptr();
        // SAFETY: `next` was loaded from `entry`'s links
        let next = unsafe { self.launder(next) };
        // SAFETY: `next` is dereferenceable (see `DerefPtr`).
        unsafe { (*next.as_ptr()).prev = prev };
        unsafe { (*entry.as_ptr()).next = EntryPtr::dangling() };
        unsafe { (*entry.as_ptr()).prev = EntryPtr::dangling() };
    }
}
