// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use core::cmp;
use core::mem;
use core::ops::Drop;
use core::ptr::{self, Unique};
use core::slice;
use heap::{Alloc, Layout, Heap};
use super::boxed::Box;

/// A low-level utility for more ergonomically allocating, reallocating, and deallocating
/// a buffer of memory on the heap without having to worry about all the corner cases
/// involved. This type is excellent for building your own data structures like Vec and VecDeque.
/// In particular:
///
/// * Produces Unique::empty() on zero-sized types
/// * Produces Unique::empty() on zero-length allocations
/// * Catches all overflows in capacity computations (promotes them to "capacity overflow" panics)
/// * Guards against 32-bit systems allocating more than isize::MAX bytes
/// * Guards against overflowing your length
/// * Aborts on OOM
/// * Avoids freeing Unique::empty()
/// * Contains a ptr::Unique and thus endows the user with all related benefits
///
/// This type does not in anyway inspect the memory that it manages. When dropped it *will*
/// free its memory, but it *won't* try to Drop its contents. It is up to the user of RawVec
/// to handle the actual things *stored* inside of a RawVec.
///
/// Note that a RawVec always forces its capacity to be usize::MAX for zero-sized types.
/// This enables you to use capacity growing logic catch the overflows in your length
/// that might occur with zero-sized types.
///
/// However this means that you need to be careful when roundtripping this type
/// with a `Box<[T]>`: `cap()` won't yield the len. However `with_capacity`,
/// `shrink_to_fit`, and `from_box` will actually set RawVec's private capacity
/// field. This allows zero-sized types to not be special-cased by consumers of
/// this type.
#[allow(missing_debug_implementations)]
pub struct RawVec<T, A: Alloc = Heap> {
    ptr: Unique<T>,
    cap: usize,
    a: A,
}

impl<T, A: Alloc> RawVec<T, A> {
    /// Like `new` but parameterized over the choice of allocator for
    /// the returned RawVec.
    pub fn new_in(a: A) -> Self {
        // !0 is usize::MAX. This branch should be stripped at compile time.
        let cap = if mem::size_of::<T>() == 0 { !0 } else { 0 };

        // Unique::empty() doubles as "unallocated" and "zero-sized allocation"
        RawVec {
            ptr: Unique::empty(),
            cap,
            a,
        }
    }

    /// Like `with_capacity` but parameterized over the choice of
    /// allocator for the returned RawVec.
    #[inline]
    pub fn with_capacity_in(cap: usize, a: A) -> Self {
        RawVec::allocate_in(cap, false, a)
    }

    /// Like `with_capacity_zeroed` but parameterized over the choice
    /// of allocator for the returned RawVec.
    #[inline]
    pub fn with_capacity_zeroed_in(cap: usize, a: A) -> Self {
        RawVec::allocate_in(cap, true, a)
    }

    fn allocate_in(cap: usize, zeroed: bool, mut a: A) -> Self {
        unsafe {
            let elem_size = mem::size_of::<T>();

            let alloc_size = cap.checked_mul(elem_size).expect("capacity overflow");
            alloc_guard(alloc_size);

            // handles ZSTs and `cap = 0` alike
            let ptr = if alloc_size == 0 {
                mem::align_of::<T>() as *mut u8
            } else {
                let align = mem::align_of::<T>();
                let result = if zeroed {
                    a.alloc_zeroed(Layout::from_size_align(alloc_size, align).unwrap())
                } else {
                    a.alloc(Layout::from_size_align(alloc_size, align).unwrap())
                };
                match result {
                    Ok(ptr) => ptr,
                    Err(err) => a.oom(err),
                }
            };

            RawVec {
                ptr: Unique::new_unchecked(ptr as *mut _),
                cap,
                a,
            }
        }
    }
}

impl<T> RawVec<T, Heap> {
    /// Creates the biggest possible RawVec (on the system heap)
    /// without allocating. If T has positive size, then this makes a
    /// RawVec with capacity 0. If T has 0 size, then it it makes a
    /// RawVec with capacity `usize::MAX`. Useful for implementing
    /// delayed allocation.
    pub fn new() -> Self {
        Self::new_in(Heap)
    }

    /// Creates a RawVec (on the system heap) with exactly the
    /// capacity and alignment requirements for a `[T; cap]`. This is
    /// equivalent to calling RawVec::new when `cap` is 0 or T is
    /// zero-sized. Note that if `T` is zero-sized this means you will
    /// *not* get a RawVec with the requested capacity!
    ///
    /// # Panics
    ///
    /// * Panics if the requested capacity exceeds `usize::MAX` bytes.
    /// * Panics on 32-bit platforms if the requested capacity exceeds
    ///   `isize::MAX` bytes.
    ///
    /// # Aborts
    ///
    /// Aborts on OOM
    #[inline]
    pub fn with_capacity(cap: usize) -> Self {
        RawVec::allocate_in(cap, false, Heap)
    }

    /// Like `with_capacity` but guarantees the buffer is zeroed.
    #[inline]
    pub fn with_capacity_zeroed(cap: usize) -> Self {
        RawVec::allocate_in(cap, true, Heap)
    }
}

impl<T, A: Alloc> RawVec<T, A> {
    /// Reconstitutes a RawVec from a pointer, capacity, and allocator.
    ///
    /// # Undefined Behavior
    ///
    /// The ptr must be allocated (via the given allocator `a`), and with the given capacity. The
    /// capacity cannot exceed `isize::MAX` (only a concern on 32-bit systems).
    /// If the ptr and capacity come from a RawVec created via `a`, then this is guaranteed.
    pub unsafe fn from_raw_parts_in(ptr: *mut T, cap: usize, a: A) -> Self {
        RawVec {
            ptr: Unique::new_unchecked(ptr),
            cap,
            a,
        }
    }
}

impl<T> RawVec<T, Heap> {
    /// Reconstitutes a RawVec from a pointer, capacity.
    ///
    /// # Undefined Behavior
    ///
    /// The ptr must be allocated (on the system heap), and with the given capacity. The
    /// capacity cannot exceed `isize::MAX` (only a concern on 32-bit systems).
    /// If the ptr and capacity come from a RawVec, then this is guaranteed.
    pub unsafe fn from_raw_parts(ptr: *mut T, cap: usize) -> Self {
        RawVec {
            ptr: Unique::new_unchecked(ptr),
            cap,
            a: Heap,
        }
    }

    /// Converts a `Box<[T]>` into a `RawVec<T>`.
    pub fn from_box(mut slice: Box<[T]>) -> Self {
        unsafe {
            let result = RawVec::from_raw_parts(slice.as_mut_ptr(), slice.len());
            mem::forget(slice);
            result
        }
    }
}

impl<T, A: Alloc> RawVec<T, A> {
    /// Gets a raw pointer to the start of the allocation. Note that this is
    /// Unique::empty() if `cap = 0` or T is zero-sized. In the former case, you must
    /// be careful.
    pub fn ptr(&self) -> *mut T {
        self.ptr.as_ptr()
    }

    /// Gets the capacity of the allocation.
    ///
    /// This will always be `usize::MAX` if `T` is zero-sized.
    #[inline(always)]
    pub fn cap(&self) -> usize {
        if mem::size_of::<T>() == 0 {
            !0
        } else {
            self.cap
        }
    }

    /// Returns a shared reference to the allocator backing this RawVec.
    pub fn alloc(&self) -> &A {
        &self.a
    }

    /// Returns a mutable reference to the allocator backing this RawVec.
    pub fn alloc_mut(&mut self) -> &mut A {
        &mut self.a
    }

    fn current_layout(&self) -> Option<Layout> {
        if self.cap == 0 {
            None
        } else {
            // We have an allocated chunk of memory, so we can bypass runtime
            // checks to get our current layout.
            unsafe {
                let align = mem::align_of::<T>();
                let size = mem::size_of::<T>() * self.cap;
                Some(Layout::from_size_align_unchecked(size, align))
            }
        }
    }

    /// Doubles the size of the type's backing allocation. This is common enough
    /// to want to do that it's easiest to just have a dedicated method. Slightly
    /// more efficient logic can be provided for this than the general case.
    ///
    /// This function is ideal for when pushing elements one-at-a-time because
    /// you don't need to incur the costs of the more general computations
    /// reserve needs to do to guard against overflow. You do however need to
    /// manually check if your `len == cap`.
    ///
    /// # Panics
    ///
    /// * Panics if T is zero-sized on the assumption that you managed to exhaust
    ///   all `usize::MAX` slots in your imaginary buffer.
    /// * Panics on 32-bit platforms if the requested capacity exceeds
    ///   `isize::MAX` bytes.
    ///
    /// # Aborts
    ///
    /// Aborts on OOM
    ///
    /// # Examples
    ///
    /// ```
    /// # #![feature(alloc)]
    /// # extern crate alloc;
    /// # use std::ptr;
    /// # use alloc::raw_vec::RawVec;
    /// struct MyVec<T> {
    ///     buf: RawVec<T>,
    ///     len: usize,
    /// }
    ///
    /// impl<T> MyVec<T> {
    ///     pub fn push(&mut self, elem: T) {
    ///         if self.len == self.buf.cap() { self.buf.double(); }
    ///         // double would have aborted or panicked if the len exceeded
    ///         // `isize::MAX` so this is safe to do unchecked now.
    ///         unsafe {
    ///             ptr::write(self.buf.ptr().offset(self.len as isize), elem);
    ///         }
    ///         self.len += 1;
    ///     }
    /// }
    /// # fn main() {
    /// #   let mut vec = MyVec { buf: RawVec::new(), len: 0 };
    /// #   vec.push(1);
    /// # }
    /// ```
    #[inline(never)]
    #[cold]
    pub fn double(&mut self) {
        unsafe {
            let elem_size = mem::size_of::<T>();

            // since we set the capacity to usize::MAX when elem_size is
            // 0, getting to here necessarily means the RawVec is overfull.
            assert!(elem_size != 0, "capacity overflow");

            let (new_cap, uniq) = match self.current_layout() {
                Some(cur) => {
                    // Since we guarantee that we never allocate more than
                    // isize::MAX bytes, `elem_size * self.cap <= isize::MAX` as
                    // a precondition, so this can't overflow. Additionally the
                    // alignment will never be too large as to "not be
                    // satisfiable", so `Layout::from_size_align` will always
                    // return `Some`.
                    //
                    // tl;dr; we bypass runtime checks due to dynamic assertions
                    // in this module, allowing us to use
                    // `from_size_align_unchecked`.
                    let new_cap = 2 * self.cap;
                    let new_size = new_cap * elem_size;
                    let new_layout = Layout::from_size_align_unchecked(new_size, cur.align());
                    alloc_guard(new_size);
                    let ptr_res = self.a.realloc(self.ptr.as_ptr() as *mut u8,
                                                 cur,
                                                 new_layout);
                    match ptr_res {
                        Ok(ptr) => (new_cap, Unique::new_unchecked(ptr as *mut T)),
                        Err(e) => self.a.oom(e),
                    }
                }
                None => {
                    // skip to 4 because tiny Vec's are dumb; but not if that
                    // would cause overflow
                    let new_cap = if elem_size > (!0) / 8 { 1 } else { 4 };
                    match self.a.alloc_array::<T>(new_cap) {
                        Ok(ptr) => (new_cap, ptr),
                        Err(e) => self.a.oom(e),
                    }
                }
            };
            self.ptr = uniq;
            self.cap = new_cap;
        }
    }

    /// Attempts to double the size of the type's backing allocation in place. This is common
    /// enough to want to do that it's easiest to just have a dedicated method. Slightly
    /// more efficient logic can be provided for this than the general case.
    ///
    /// Returns true if the reallocation attempt has succeeded, or false otherwise.
    ///
    /// # Panics
    ///
    /// * Panics if T is zero-sized on the assumption that you managed to exhaust
    ///   all `usize::MAX` slots in your imaginary buffer.
    /// * Panics on 32-bit platforms if the requested capacity exceeds
    ///   `isize::MAX` bytes.
    #[inline(never)]
    #[cold]
    pub fn double_in_place(&mut self) -> bool {
        unsafe {
            let elem_size = mem::size_of::<T>();
            let old_layout = match self.current_layout() {
                Some(layout) => layout,
                None => return false, // nothing to double
            };

            // since we set the capacity to usize::MAX when elem_size is
            // 0, getting to here necessarily means the RawVec is overfull.
            assert!(elem_size != 0, "capacity overflow");

            // Since we guarantee that we never allocate more than isize::MAX
            // bytes, `elem_size * self.cap <= isize::MAX` as a precondition, so
            // this can't overflow.
            //
            // Similarly like with `double` above we can go straight to
            // `Layout::from_size_align_unchecked` as we know this won't
            // overflow and the alignment is sufficiently small.
            let new_cap = 2 * self.cap;
            let new_size = new_cap * elem_size;
            alloc_guard(new_size);
            let ptr = self.ptr() as *mut _;
            let new_layout = Layout::from_size_align_unchecked(new_size, old_layout.align());
            match self.a.grow_in_place(ptr, old_layout, new_layout) {
                Ok(_) => {
                    // We can't directly divide `size`.
                    self.cap = new_cap;
                    true
                }
                Err(_) => {
                    false
                }
            }
        }
    }

    /// Ensures that the buffer contains at least enough space to hold
    /// `used_cap + needed_extra_cap` elements. If it doesn't already,
    /// will reallocate the minimum possible amount of memory necessary.
    /// Generally this will be exactly the amount of memory necessary,
    /// but in principle the allocator is free to give back more than
    /// we asked for.
    ///
    /// If `used_cap` exceeds `self.cap()`, this may fail to actually allocate
    /// the requested space. This is not really unsafe, but the unsafe
    /// code *you* write that relies on the behavior of this function may break.
    ///
    /// # Panics
    ///
    /// * Panics if the requested capacity exceeds `usize::MAX` bytes.
    /// * Panics on 32-bit platforms if the requested capacity exceeds
    ///   `isize::MAX` bytes.
    ///
    /// # Aborts
    ///
    /// Aborts on OOM
    pub fn reserve_exact(&mut self, used_cap: usize, needed_extra_cap: usize) {
        unsafe {
            // NOTE: we don't early branch on ZSTs here because we want this
            // to actually catch "asking for more than usize::MAX" in that case.
            // If we make it past the first branch then we are guaranteed to
            // panic.

            // Don't actually need any more capacity.
            // Wrapping in case they gave a bad `used_cap`.
            if self.cap().wrapping_sub(used_cap) >= needed_extra_cap {
                return;
            }

            // Nothing we can really do about these checks :(
            let new_cap = used_cap.checked_add(needed_extra_cap).expect("capacity overflow");
            let new_layout = match Layout::array::<T>(new_cap) {
                Some(layout) => layout,
                None => panic!("capacity overflow"),
            };
            alloc_guard(new_layout.size());
            let res = match self.current_layout() {
                Some(layout) => {
                    let old_ptr = self.ptr.as_ptr() as *mut u8;
                    self.a.realloc(old_ptr, layout, new_layout)
                }
                None => self.a.alloc(new_layout),
            };
            let uniq = match res {
                Ok(ptr) => Unique::new_unchecked(ptr as *mut T),
                Err(e) => self.a.oom(e),
            };
            self.ptr = uniq;
            self.cap = new_cap;
        }
    }

    /// Calculates the buffer's new size given that it'll hold `used_cap +
    /// needed_extra_cap` elements. This logic is used in amortized reserve methods.
    /// Returns `(new_capacity, new_alloc_size)`.
    fn amortized_new_size(&self, used_cap: usize, needed_extra_cap: usize) -> usize {
        // Nothing we can really do about these checks :(
        let required_cap = used_cap.checked_add(needed_extra_cap)
            .expect("capacity overflow");
        // Cannot overflow, because `cap <= isize::MAX`, and type of `cap` is `usize`.
        let double_cap = self.cap * 2;
        // `double_cap` guarantees exponential growth.
        cmp::max(double_cap, required_cap)
    }

    /// Ensures that the buffer contains at least enough space to hold
    /// `used_cap + needed_extra_cap` elements. If it doesn't already have
    /// enough capacity, will reallocate enough space plus comfortable slack
    /// space to get amortized `O(1)` behavior. Will limit this behavior
    /// if it would needlessly cause itself to panic.
    ///
    /// If `used_cap` exceeds `self.cap()`, this may fail to actually allocate
    /// the requested space. This is not really unsafe, but the unsafe
    /// code *you* write that relies on the behavior of this function may break.
    ///
    /// This is ideal for implementing a bulk-push operation like `extend`.
    ///
    /// # Panics
    ///
    /// * Panics if the requested capacity exceeds `usize::MAX` bytes.
    /// * Panics on 32-bit platforms if the requested capacity exceeds
    ///   `isize::MAX` bytes.
    ///
    /// # Aborts
    ///
    /// Aborts on OOM
    ///
    /// # Examples
    ///
    /// ```
    /// # #![feature(alloc)]
    /// # extern crate alloc;
    /// # use std::ptr;
    /// # use alloc::raw_vec::RawVec;
    /// struct MyVec<T> {
    ///     buf: RawVec<T>,
    ///     len: usize,
    /// }
    ///
    /// impl<T: Clone> MyVec<T> {
    ///     pub fn push_all(&mut self, elems: &[T]) {
    ///         self.buf.reserve(self.len, elems.len());
    ///         // reserve would have aborted or panicked if the len exceeded
    ///         // `isize::MAX` so this is safe to do unchecked now.
    ///         for x in elems {
    ///             unsafe {
    ///                 ptr::write(self.buf.ptr().offset(self.len as isize), x.clone());
    ///             }
    ///             self.len += 1;
    ///         }
    ///     }
    /// }
    /// # fn main() {
    /// #   let mut vector = MyVec { buf: RawVec::new(), len: 0 };
    /// #   vector.push_all(&[1, 3, 5, 7, 9]);
    /// # }
    /// ```
    pub fn reserve(&mut self, used_cap: usize, needed_extra_cap: usize) {
        unsafe {
            // NOTE: we don't early branch on ZSTs here because we want this
            // to actually catch "asking for more than usize::MAX" in that case.
            // If we make it past the first branch then we are guaranteed to
            // panic.

            // Don't actually need any more capacity.
            // Wrapping in case they give a bad `used_cap`
            if self.cap().wrapping_sub(used_cap) >= needed_extra_cap {
                return;
            }

            let new_cap = self.amortized_new_size(used_cap, needed_extra_cap);

            let new_layout = match Layout::array::<T>(new_cap) {
                Some(layout) => layout,
                None => panic!("capacity overflow"),
            };
            // FIXME: may crash and burn on over-reserve
            alloc_guard(new_layout.size());
            let res = match self.current_layout() {
                Some(layout) => {
                    let old_ptr = self.ptr.as_ptr() as *mut u8;
                    self.a.realloc(old_ptr, layout, new_layout)
                }
                None => self.a.alloc(new_layout),
            };
            let uniq = match res {
                Ok(ptr) => Unique::new_unchecked(ptr as *mut T),
                Err(e) => self.a.oom(e),
            };
            self.ptr = uniq;
            self.cap = new_cap;
        }
    }

    /// Attempts to ensure that the buffer contains at least enough space to hold
    /// `used_cap + needed_extra_cap` elements. If it doesn't already have
    /// enough capacity, will reallocate in place enough space plus comfortable slack
    /// space to get amortized `O(1)` behavior. Will limit this behaviour
    /// if it would needlessly cause itself to panic.
    ///
    /// If `used_cap` exceeds `self.cap()`, this may fail to actually allocate
    /// the requested space. This is not really unsafe, but the unsafe
    /// code *you* write that relies on the behavior of this function may break.
    ///
    /// Returns true if the reallocation attempt has succeeded, or false otherwise.
    ///
    /// # Panics
    ///
    /// * Panics if the requested capacity exceeds `usize::MAX` bytes.
    /// * Panics on 32-bit platforms if the requested capacity exceeds
    ///   `isize::MAX` bytes.
    pub fn reserve_in_place(&mut self, used_cap: usize, needed_extra_cap: usize) -> bool {
        unsafe {
            // NOTE: we don't early branch on ZSTs here because we want this
            // to actually catch "asking for more than usize::MAX" in that case.
            // If we make it past the first branch then we are guaranteed to
            // panic.

            // Don't actually need any more capacity. If the current `cap` is 0, we can't
            // reallocate in place.
            // Wrapping in case they give a bad `used_cap`
            let old_layout = match self.current_layout() {
                Some(layout) => layout,
                None => return false,
            };
            if self.cap().wrapping_sub(used_cap) >= needed_extra_cap {
                return false;
            }

            let new_cap = self.amortized_new_size(used_cap, needed_extra_cap);

            // Here, `cap < used_cap + needed_extra_cap <= new_cap`
            // (regardless of whether `self.cap - used_cap` wrapped).
            // Therefore we can safely call grow_in_place.

            let ptr = self.ptr() as *mut _;
            let new_layout = Layout::new::<T>().repeat(new_cap).unwrap().0;
            // FIXME: may crash and burn on over-reserve
            alloc_guard(new_layout.size());
            match self.a.grow_in_place(ptr, old_layout, new_layout) {
                Ok(_) => {
                    self.cap = new_cap;
                    true
                }
                Err(_) => {
                    false
                }
            }
        }
    }

    /// Shrinks the allocation down to the specified amount. If the given amount
    /// is 0, actually completely deallocates.
    ///
    /// # Panics
    ///
    /// Panics if the given amount is *larger* than the current capacity.
    ///
    /// # Aborts
    ///
    /// Aborts on OOM.
    pub fn shrink_to_fit(&mut self, amount: usize) {
        let elem_size = mem::size_of::<T>();

        // Set the `cap` because they might be about to promote to a `Box<[T]>`
        if elem_size == 0 {
            self.cap = amount;
            return;
        }

        // This check is my waterloo; it's the only thing Vec wouldn't have to do.
        assert!(self.cap >= amount, "Tried to shrink to a larger capacity");

        if amount == 0 {
            // We want to create a new zero-length vector within the
            // same allocator.  We use ptr::write to avoid an
            // erroneous attempt to drop the contents, and we use
            // ptr::read to sidestep condition against destructuring
            // types that implement Drop.

            unsafe {
                let a = ptr::read(&self.a as *const A);
                self.dealloc_buffer();
                ptr::write(self, RawVec::new_in(a));
            }
        } else if self.cap != amount {
            unsafe {
                // We know here that our `amount` is greater than zero. This
                // implies, via the assert above, that capacity is also greater
                // than zero, which means that we've got a current layout that
                // "fits"
                //
                // We also know that `self.cap` is greater than `amount`, and
                // consequently we don't need runtime checks for creating either
                // layout
                let old_size = elem_size * self.cap;
                let new_size = elem_size * amount;
                let align = mem::align_of::<T>();
                let old_layout = Layout::from_size_align_unchecked(old_size, align);
                let new_layout = Layout::from_size_align_unchecked(new_size, align);
                match self.a.realloc(self.ptr.as_ptr() as *mut u8,
                                     old_layout,
                                     new_layout) {
                    Ok(p) => self.ptr = Unique::new_unchecked(p as *mut T),
                    Err(err) => self.a.oom(err),
                }
            }
            self.cap = amount;
        }
    }
}

impl<T> RawVec<T, Heap> {
    /// Converts the entire buffer into `Box<[T]>`.
    ///
    /// While it is not *strictly* Undefined Behavior to call
    /// this procedure while some of the RawVec is uninitialized,
    /// it certainly makes it trivial to trigger it.
    ///
    /// Note that this will correctly reconstitute any `cap` changes
    /// that may have been performed. (see description of type for details)
    pub unsafe fn into_box(self) -> Box<[T]> {
        // NOTE: not calling `cap()` here, actually using the real `cap` field!
        let slice = slice::from_raw_parts_mut(self.ptr(), self.cap);
        let output: Box<[T]> = Box::from_raw(slice);
        mem::forget(self);
        output
    }
}

impl<T, A: Alloc> RawVec<T, A> {
    /// Frees the memory owned by the RawVec *without* trying to Drop its contents.
    pub unsafe fn dealloc_buffer(&mut self) {
        let elem_size = mem::size_of::<T>();
        if elem_size != 0 {
            if let Some(layout) = self.current_layout() {
                let ptr = self.ptr() as *mut u8;
                self.a.dealloc(ptr, layout);
            }
        }
    }
}

unsafe impl<#[may_dangle] T, A: Alloc> Drop for RawVec<T, A> {
    /// Frees the memory owned by the RawVec *without* trying to Drop its contents.
    fn drop(&mut self) {
        unsafe { self.dealloc_buffer(); }
    }
}



// We need to guarantee the following:
// * We don't ever allocate `> isize::MAX` byte-size objects
// * We don't overflow `usize::MAX` and actually allocate too little
//
// On 64-bit we just need to check for overflow since trying to allocate
// `> isize::MAX` bytes will surely fail. On 32-bit and 16-bit we need to add
// an extra guard for this in case we're running on a platform which can use
// all 4GB in user-space. e.g. PAE or x32

#[inline]
fn alloc_guard(alloc_size: usize) {
    if mem::size_of::<usize>() < 8 {
        assert!(alloc_size <= ::core::isize::MAX as usize,
                "capacity overflow");
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allocator_param() {
        use allocator::{Alloc, AllocErr};

        // Writing a test of integration between third-party
        // allocators and RawVec is a little tricky because the RawVec
        // API does not expose fallible allocation methods, so we
        // cannot check what happens when allocator is exhausted
        // (beyond detecting a panic).
        //
        // Instead, this just checks that the RawVec methods do at
        // least go through the Allocator API when it reserves
        // storage.

        // A dumb allocator that consumes a fixed amount of fuel
        // before allocation attempts start failing.
        struct BoundedAlloc { fuel: usize }
        unsafe impl Alloc for BoundedAlloc {
            unsafe fn alloc(&mut self, layout: Layout) -> Result<*mut u8, AllocErr> {
                let size = layout.size();
                if size > self.fuel {
                    return Err(AllocErr::Unsupported { details: "fuel exhausted" });
                }
                match Heap.alloc(layout) {
                    ok @ Ok(_) => { self.fuel -= size; ok }
                    err @ Err(_) => err,
                }
            }
            unsafe fn dealloc(&mut self, ptr: *mut u8, layout: Layout) {
                Heap.dealloc(ptr, layout)
            }
        }

        let a = BoundedAlloc { fuel: 500 };
        let mut v: RawVec<u8, _> = RawVec::with_capacity_in(50, a);
        assert_eq!(v.a.fuel, 450);
        v.reserve(50, 150); // (causes a realloc, thus using 50 + 150 = 200 units of fuel)
        assert_eq!(v.a.fuel, 250);
    }

    #[test]
    fn reserve_does_not_overallocate() {
        {
            let mut v: RawVec<u32> = RawVec::new();
            // First `reserve` allocates like `reserve_exact`
            v.reserve(0, 9);
            assert_eq!(9, v.cap());
        }

        {
            let mut v: RawVec<u32> = RawVec::new();
            v.reserve(0, 7);
            assert_eq!(7, v.cap());
            // 97 if more than double of 7, so `reserve` should work
            // like `reserve_exact`.
            v.reserve(7, 90);
            assert_eq!(97, v.cap());
        }

        {
            let mut v: RawVec<u32> = RawVec::new();
            v.reserve(0, 12);
            assert_eq!(12, v.cap());
            v.reserve(12, 3);
            // 3 is less than half of 12, so `reserve` must grow
            // exponentially. At the time of writing this test grow
            // factor is 2, so new capacity is 24, however, grow factor
            // of 1.5 is OK too. Hence `>= 18` in assert.
            assert!(v.cap() >= 12 + 12 / 2);
        }
    }


}
