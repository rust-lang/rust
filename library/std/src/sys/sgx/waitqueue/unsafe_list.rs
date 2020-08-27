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

pub struct UnsafeList<T> {
    head_tail: NonNull<UnsafeListEntry<T>>,
    head_tail_entry: Option<UnsafeListEntry<T>>,
}

impl<T> UnsafeList<T> {
    pub const fn new() -> Self {
        unsafe { UnsafeList { head_tail: NonNull::new_unchecked(1 as _), head_tail_entry: None } }
    }

    unsafe fn init(&mut self) {
        if self.head_tail_entry.is_none() {
            self.head_tail_entry = Some(UnsafeListEntry::dummy());
            self.head_tail = NonNull::new_unchecked(self.head_tail_entry.as_mut().unwrap());
            self.head_tail.as_mut().next = self.head_tail;
            self.head_tail.as_mut().prev = self.head_tail;
        }
    }

    pub fn is_empty(&self) -> bool {
        unsafe {
            if self.head_tail_entry.is_some() {
                let first = self.head_tail.as_ref().next;
                if first == self.head_tail {
                    // ,-------> /---------\ next ---,
                    // |         |head_tail|         |
                    // `--- prev \---------/ <-------`
                    rtassert!(self.head_tail.as_ref().prev == first);
                    true
                } else {
                    false
                }
            } else {
                true
            }
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
        let mut entry = NonNull::new_unchecked(entry);
        let mut prev_tail = mem::replace(&mut self.head_tail.as_mut().prev, entry);
        entry.as_mut().prev = prev_tail;
        entry.as_mut().next = self.head_tail;
        prev_tail.as_mut().next = entry;
        // unwrap ok: always `Some` on non-dummy entries
        (*entry.as_ptr()).value.as_ref().unwrap()
    }

    /// Pops an entry from the front of the list.
    ///
    /// # Safety
    ///
    /// The caller must make sure to synchronize ending the borrow of the
    /// return value and deallocation of the containing entry.
    pub unsafe fn pop<'a>(&mut self) -> Option<&'a T> {
        self.init();

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
            let mut first = self.head_tail.as_mut().next;
            let mut second = first.as_mut().next;
            self.head_tail.as_mut().next = second;
            second.as_mut().prev = self.head_tail;
            first.as_mut().next = NonNull::dangling();
            first.as_mut().prev = NonNull::dangling();
            // unwrap ok: always `Some` on non-dummy entries
            Some((*first.as_ptr()).value.as_ref().unwrap())
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
        prev.as_mut().next = next;
        next.as_mut().prev = prev;
        entry.next = NonNull::dangling();
        entry.prev = NonNull::dangling();
    }
}
