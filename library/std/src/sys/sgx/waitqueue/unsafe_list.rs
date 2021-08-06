//! A doubly-linked list where callers are in charge of memory allocation
//! of the nodes in the list.

#[cfg(test)]
mod tests;

use crate::mem;
use crate::ptr::NonNull;

pub struct UnsafeListEntry<T> {
    next: NonNull<UnsafeListEntry<T>>,
    prev: NonNull<UnsafeListEntry<T>>,
    value: Option<T>,
}

impl<T> UnsafeListEntry<T> {
    fn dummy() -> Self {
        UnsafeListEntry { next: NonNull::dangling(), prev: NonNull::dangling(), value: None }
    }

    pub fn new(value: T) -> Self {
        UnsafeListEntry { value: Some(value), ..Self::dummy() }
    }
}

// WARNING: self-referential struct!
pub struct UnsafeList<T> {
    head_tail: NonNull<UnsafeListEntry<T>>,
    head_tail_entry: Option<UnsafeListEntry<T>>,
}

impl<T> UnsafeList<T> {
    pub const fn new() -> Self {
        unsafe { UnsafeList { head_tail: NonNull::new_unchecked(1 as _), head_tail_entry: None } }
    }

    /// # Safety
    unsafe fn init(&mut self) {
        if self.head_tail_entry.is_none() {
            self.head_tail_entry = Some(UnsafeListEntry::dummy());
            // SAFETY: `head_tail_entry` must be non-null, which it is because we assign it above.
            self.head_tail =
                unsafe { NonNull::new_unchecked(self.head_tail_entry.as_mut().unwrap()) };
            // SAFETY: `self.head_tail` must meet all requirements for a mutable reference.
            unsafe { self.head_tail.as_mut() }.next = self.head_tail;
            unsafe { self.head_tail.as_mut() }.prev = self.head_tail;
        }
    }

    pub fn is_empty(&self) -> bool {
        if self.head_tail_entry.is_some() {
            let first = unsafe { self.head_tail.as_ref() }.next;
            if first == self.head_tail {
                // ,-------> /---------\ next ---,
                // |         |head_tail|         |
                // `--- prev \---------/ <-------`
                // SAFETY: `self.head_tail` must meet all requirements for a reference.
                unsafe { rtassert!(self.head_tail.as_ref().prev == first) };
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
    /// not destroy the stack frame containing the entry.
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
        let mut entry = unsafe { NonNull::new_unchecked(entry) };
        let mut prev_tail = mem::replace(&mut unsafe { self.head_tail.as_mut() }.prev, entry);
        // SAFETY: `entry` must meet all requirements for a mutable reference.
        unsafe { entry.as_mut() }.prev = prev_tail;
        unsafe { entry.as_mut() }.next = self.head_tail;
        // SAFETY: `prev_tail` must meet all requirements for a mutable reference.
        unsafe { prev_tail.as_mut() }.next = entry;
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
        unsafe { self.init() };

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
            let mut first = unsafe { self.head_tail.as_mut() }.next;
            let mut second = unsafe { first.as_mut() }.next;
            unsafe { self.head_tail.as_mut() }.next = second;
            unsafe { second.as_mut() }.prev = self.head_tail;
            unsafe { first.as_mut() }.next = NonNull::dangling();
            unsafe { first.as_mut() }.prev = NonNull::dangling();
            // unwrap ok: always `Some` on non-dummy entries
            Some(unsafe { (*first.as_ptr()).value.as_ref() }.unwrap())
        }
    }

    /// Removes an entry from the list.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `entry` has been pushed onto `self`
    /// prior to this call and has not moved since then.
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
        let mut prev = entry.prev;
        let mut next = entry.next;
        // SAFETY: `prev` and `next` must meet all requirements for a mutable reference.entry
        unsafe { prev.as_mut() }.next = next;
        unsafe { next.as_mut() }.prev = prev;
        entry.next = NonNull::dangling();
        entry.prev = NonNull::dangling();
    }
}
