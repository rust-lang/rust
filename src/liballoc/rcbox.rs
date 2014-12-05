// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use core::atomic;
use core::cell::UnsafeCell;
use core::mem;
use core::ptr;

/// A shared representation for reference counted pointers. This is factored out
/// to allow upgrading from one to another seamlessly.
pub struct RcBox<T> {
    strong: UnsafeCell<uint>,
    weak:   UnsafeCell<uint>,
    value:  T,
}

/// The result of decrementing a refcount.
#[must_use]
pub enum DecResult {
    /// The caller must call deallocate.
    NeedsFree,
    /// There's still strong (or weak) pointers outstanding.
    StillInUse
}

impl<T> RcBox<T> {
    #[inline]
    pub fn new(value: T) -> RcBox<T> {
        RcBox {
            strong: UnsafeCell::new(1),
            // there is an implicit weak pointer owned by all the
            // strong pointers, which ensures that the weak
            // destructor never frees the allocation while the
            // strong destructor is running, even if the weak
            // pointer is stored inside the strong one.
            weak: UnsafeCell::new(1),
            value: value,
        }
    }

    #[inline(always)]
    pub fn strong_nonatomic(&self) -> uint {
        unsafe { *self.strong.get() }
    }

    #[inline(always)]
    fn set_strong_nonatomic(&self, x: uint) {
        unsafe { *self.strong.get() = x; }
    }

    #[inline(always)]
    pub fn weak_nonatomic(&self) -> uint {
        unsafe { *self.weak.get() }
    }

    #[inline(always)]
    fn set_weak_nonatomic(&self, x: uint) {
        unsafe { *self.weak.get() = x; }
    }

    #[inline]
    pub fn inc_strong_nonatomic(&self) {
        let old_strong = self.strong_nonatomic();
        self.set_strong_nonatomic(old_strong + 1);
    }

    #[inline]
    pub fn dec_strong_nonatomic(&self) -> DecResult {
        let old_strong = self.strong_nonatomic();
        debug_assert!(old_strong != 0);
        self.set_strong_nonatomic(old_strong - 1);
        if old_strong == 1 {
            unsafe { ptr::read(self.value()); } // destroy the contained object
            DecResult::NeedsFree
        } else {
            DecResult::StillInUse
        }
    }

    #[inline]
    pub fn inc_weak_nonatomic(&self) {
        let old_weak = self.weak_nonatomic();
        self.set_weak_nonatomic(old_weak + 1);
    }

    #[inline]
    pub fn dec_weak_nonatomic(&self) -> DecResult {
        let old_weak = self.weak_nonatomic();
        debug_assert!(old_weak != 0);
        self.set_weak_nonatomic(old_weak - 1);
        if old_weak == 1 {
            DecResult::NeedsFree
        } else {
            DecResult::StillInUse
        }
    }


    #[inline(always)]
    pub unsafe fn strong_atomic<'a>(&'a self) -> &'a atomic::AtomicUint {
        let strong_ref: &uint = &*self.strong.get();
        mem::transmute(strong_ref)
    }

    #[inline(always)]
    pub unsafe fn weak_atomic<'a>(&'a self) -> &'a atomic::AtomicUint {
        let weak_ref: &uint = &*self.weak.get();
        mem::transmute(weak_ref)
    }

    #[inline]
    pub fn inc_strong_atomic(&self) {
        // Using a relaxed ordering is alright here, as knowledge of the
        // original reference prevents other threads from erroneously deleting
        // the object.
        //
        // As explained in the [Boost documentation][1], Increasing the
        // reference counter can always be done with memory_order_relaxed: New
        // references to an object can only be formed from an existing
        // reference, and passing an existing reference from one thread to
        // another must already provide any required synchronization.
        //
        // [1]: (www.boost.org/doc/libs/1_55_0/doc/html/atomic/usage_examples.html)
        unsafe {
            self.strong_atomic().fetch_add(1, atomic::Relaxed);
        }
    }

    #[inline]
    pub fn dec_strong_atomic(&self) -> DecResult {
        unsafe {
            // Because `fetch_sub` is already atomic, we do not need to synchronize
            // with other threads unless we are going to delete the object. This
            // same logic applies to the below `fetch_sub` to the `weak` count.
            if self.strong_atomic().fetch_sub(1, atomic::Release) != 1 {
                return DecResult::StillInUse
            }

            // This fence is needed to prevent reordering of use of the data and
            // deletion of the data. Because it is marked `Release`, the
            // decreasing of the reference count synchronizes with this `Acquire`
            // fence. This means that use of the data happens before decreasing
            // the reference count, which happens before this fence, which
            // happens before the deletion of the data.
            //
            // As explained in the [Boost documentation][1],
            //
            // It is important to enforce any possible access to the object in
            // one thread (through an existing reference) to *happen before*
            // deleting the object in a different thread. This is achieved by a
            // "release" operation after dropping a reference (any access to the
            // object through this reference must obviously happened before),
            // and an "acquire" operation before deleting the object.
            //
            // [1]: (www.boost.org/doc/libs/1_55_0/doc/html/atomic/usage_examples.html)
            atomic::fence(atomic::Acquire);

            // Destroy the data at this time, even though we may not free the box
            // allocation itself (there may still be weak pointers lying around).
            mem::drop(ptr::read(self.value()));

            DecResult::NeedsFree
        }
    }

    #[inline]
    pub fn inc_weak_atomic(&self) {
        // See comments in RcBox::inc_strong_atomic() for why this is relaxed
        unsafe {
            self.weak_atomic().fetch_add(1, atomic::Relaxed);
        }
    }

    #[inline]
    pub fn dec_weak_atomic(&self) -> DecResult {
        unsafe {
            if self.weak_atomic().fetch_sub(1, atomic::Release) == 1 {
                atomic::fence(atomic::Acquire);
                DecResult::NeedsFree
            } else {
                DecResult::StillInUse
            }
        }
    }

    #[inline(always)]
    pub fn value(&self) -> &T {
        &self.value
    }

    #[inline(always)]
    pub fn value_mut(&mut self) -> &mut T {
        &mut self.value
    }
}
