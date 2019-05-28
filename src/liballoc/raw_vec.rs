#![unstable(feature = "raw_vec_internals", reason = "implementation detail", issue = "0")]
#![doc(hidden)]

use core::cmp;
use core::mem;
use core::ops::Drop;
use core::ptr::{self, NonNull, Unique};
use core::slice;

use crate::alloc::{Alloc, Layout, Global, handle_alloc_error};
use crate::collections::CollectionAllocErr::{self, *};
use crate::boxed::Box;

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
/// * Aborts on OOM or calls handle_alloc_error as applicable
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
/// However this means that you need to be careful when round-tripping this type
/// with a `Box<[T]>`: `cap()` won't yield the len. However `with_capacity`,
/// `shrink_to_fit`, and `from_box` will actually set RawVec's private capacity
/// field. This allows zero-sized types to not be special-cased by consumers of
/// this type.
#[allow(missing_debug_implementations)]
pub struct RawVec<T, A: Alloc = Global> {
    ptr: Unique<T>,
    cap: usize,
    a: A,
}

impl<T, A: Alloc> RawVec<T, A> {
    /// Like `new` but parameterized over the choice of allocator for
    /// the returned RawVec.
    pub const fn new_in(a: A) -> Self {
        // !0 is usize::MAX. This branch should be stripped at compile time.
        // FIXME(mark-i-m): use this line when `if`s are allowed in `const`
        //let cap = if mem::size_of::<T>() == 0 { !0 } else { 0 };

        // Unique::empty() doubles as "unallocated" and "zero-sized allocation"
        RawVec {
            ptr: Unique::empty(),
            // FIXME(mark-i-m): use `cap` when ifs are allowed in const
            cap: [0, !0][(mem::size_of::<T>() == 0) as usize],
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

            let alloc_size = cap.checked_mul(elem_size).unwrap_or_else(|| capacity_overflow());
            alloc_guard(alloc_size).unwrap_or_else(|_| capacity_overflow());

            // handles ZSTs and `cap = 0` alike
            let ptr = if alloc_size == 0 {
                NonNull::<T>::dangling()
            } else {
                let align = mem::align_of::<T>();
                let layout = Layout::from_size_align(alloc_size, align).unwrap();
                let result = if zeroed {
                    a.alloc_zeroed(layout)
                } else {
                    a.alloc(layout)
                };
                match result {
                    Ok(ptr) => ptr.cast(),
                    Err(_) => handle_alloc_error(layout),
                }
            };

            RawVec {
                ptr: ptr.into(),
                cap,
                a,
            }
        }
    }
}

impl<T> RawVec<T, Global> {
    /// Creates the biggest possible RawVec (on the system heap)
    /// without allocating. If T has positive size, then this makes a
    /// RawVec with capacity 0. If T has 0 size, then it makes a
    /// RawVec with capacity `usize::MAX`. Useful for implementing
    /// delayed allocation.
    pub const fn new() -> Self {
        Self::new_in(Global)
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
        RawVec::allocate_in(cap, false, Global)
    }

    /// Like `with_capacity` but guarantees the buffer is zeroed.
    #[inline]
    pub fn with_capacity_zeroed(cap: usize) -> Self {
        RawVec::allocate_in(cap, true, Global)
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

impl<T> RawVec<T, Global> {
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
            a: Global,
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
    /// # #![feature(raw_vec_internals)]
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
    ///             ptr::write(self.buf.ptr().add(self.len), elem);
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
                    alloc_guard(new_size).unwrap_or_else(|_| capacity_overflow());
                    let ptr_res = self.a.realloc(NonNull::from(self.ptr).cast(),
                                                 cur,
                                                 new_size);
                    match ptr_res {
                        Ok(ptr) => (new_cap, ptr.cast().into()),
                        Err(_) => handle_alloc_error(
                            Layout::from_size_align_unchecked(new_size, cur.align())
                        ),
                    }
                }
                None => {
                    // skip to 4 because tiny Vec's are dumb; but not if that
                    // would cause overflow
                    let new_cap = if elem_size > (!0) / 8 { 1 } else { 4 };
                    match self.a.alloc_array::<T>(new_cap) {
                        Ok(ptr) => (new_cap, ptr.into()),
                        Err(_) => handle_alloc_error(Layout::array::<T>(new_cap).unwrap()),
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
    /// Returns `true` if the reallocation attempt has succeeded.
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
            alloc_guard(new_size).unwrap_or_else(|_| capacity_overflow());
            match self.a.grow_in_place(NonNull::from(self.ptr).cast(), old_layout, new_size) {
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

    /// The same as `reserve_exact`, but returns on errors instead of panicking or aborting.
    pub fn try_reserve_exact(&mut self, used_cap: usize, needed_extra_cap: usize)
           -> Result<(), CollectionAllocErr> {

        self.reserve_internal(used_cap, needed_extra_cap, Fallible, Exact)
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
        match self.reserve_internal(used_cap, needed_extra_cap, Infallible, Exact) {
            Err(CapacityOverflow) => capacity_overflow(),
            Err(AllocErr) => unreachable!(),
            Ok(()) => { /* yay */ }
         }
     }

    /// Calculates the buffer's new size given that it'll hold `used_cap +
    /// needed_extra_cap` elements. This logic is used in amortized reserve methods.
    /// Returns `(new_capacity, new_alloc_size)`.
    fn amortized_new_size(&self, used_cap: usize, needed_extra_cap: usize)
        -> Result<usize, CollectionAllocErr> {

        // Nothing we can really do about these checks :(
        let required_cap = used_cap.checked_add(needed_extra_cap).ok_or(CapacityOverflow)?;
        // Cannot overflow, because `cap <= isize::MAX`, and type of `cap` is `usize`.
        let double_cap = self.cap * 2;
        // `double_cap` guarantees exponential growth.
        Ok(cmp::max(double_cap, required_cap))
    }

    /// The same as `reserve`, but returns on errors instead of panicking or aborting.
    pub fn try_reserve(&mut self, used_cap: usize, needed_extra_cap: usize)
        -> Result<(), CollectionAllocErr> {
        self.reserve_internal(used_cap, needed_extra_cap, Fallible, Amortized)
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
    /// # #![feature(raw_vec_internals)]
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
    ///                 ptr::write(self.buf.ptr().add(self.len), x.clone());
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
        match self.reserve_internal(used_cap, needed_extra_cap, Infallible, Amortized) {
            Err(CapacityOverflow) => capacity_overflow(),
            Err(AllocErr) => unreachable!(),
            Ok(()) => { /* yay */ }
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
    /// Returns `true` if the reallocation attempt has succeeded.
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

            let new_cap = self.amortized_new_size(used_cap, needed_extra_cap)
                .unwrap_or_else(|_| capacity_overflow());

            // Here, `cap < used_cap + needed_extra_cap <= new_cap`
            // (regardless of whether `self.cap - used_cap` wrapped).
            // Therefore we can safely call grow_in_place.

            let new_layout = Layout::new::<T>().repeat(new_cap).unwrap().0;
            // FIXME: may crash and burn on over-reserve
            alloc_guard(new_layout.size()).unwrap_or_else(|_| capacity_overflow());
            match self.a.grow_in_place(
                NonNull::from(self.ptr).cast(), old_layout, new_layout.size(),
            ) {
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
                match self.a.realloc(NonNull::from(self.ptr).cast(),
                                     old_layout,
                                     new_size) {
                    Ok(p) => self.ptr = p.cast().into(),
                    Err(_) => handle_alloc_error(
                        Layout::from_size_align_unchecked(new_size, align)
                    ),
                }
            }
            self.cap = amount;
        }
    }
}

enum Fallibility {
    Fallible,
    Infallible,
}

use Fallibility::*;

enum ReserveStrategy {
    Exact,
    Amortized,
}

use ReserveStrategy::*;

impl<T, A: Alloc> RawVec<T, A> {
    fn reserve_internal(
        &mut self,
        used_cap: usize,
        needed_extra_cap: usize,
        fallibility: Fallibility,
        strategy: ReserveStrategy,
    ) -> Result<(), CollectionAllocErr> {
        unsafe {
            use crate::alloc::AllocErr;

            // NOTE: we don't early branch on ZSTs here because we want this
            // to actually catch "asking for more than usize::MAX" in that case.
            // If we make it past the first branch then we are guaranteed to
            // panic.

            // Don't actually need any more capacity.
            // Wrapping in case they gave a bad `used_cap`.
            if self.cap().wrapping_sub(used_cap) >= needed_extra_cap {
                return Ok(());
            }

            // Nothing we can really do about these checks :(
            let new_cap = match strategy {
                Exact => used_cap.checked_add(needed_extra_cap).ok_or(CapacityOverflow)?,
                Amortized => self.amortized_new_size(used_cap, needed_extra_cap)?,
            };
            let new_layout = Layout::array::<T>(new_cap).map_err(|_| CapacityOverflow)?;

            alloc_guard(new_layout.size())?;

            let res = match self.current_layout() {
                Some(layout) => {
                    debug_assert!(new_layout.align() == layout.align());
                    self.a.realloc(NonNull::from(self.ptr).cast(), layout, new_layout.size())
                }
                None => self.a.alloc(new_layout),
            };

            match (&res, fallibility) {
                (Err(AllocErr), Infallible) => handle_alloc_error(new_layout),
                _ => {}
            }

            self.ptr = res?.cast().into();
            self.cap = new_cap;

            Ok(())
        }
    }

}

impl<T> RawVec<T, Global> {
    /// Converts the entire buffer into `Box<[T]>`.
    ///
    /// Note that this will correctly reconstitute any `cap` changes
    /// that may have been performed. (see description of type for details)
    ///
    /// # Undefined Behavior
    ///
    /// All elements of `RawVec<T, Global>` must be initialized. Notice that
    /// the rules around uninitialized boxed values are not finalized yet,
    /// but until they are, it is advisable to avoid them.
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
                self.a.dealloc(NonNull::from(self.ptr).cast(), layout);
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
// all 4GB in user-space. e.g., PAE or x32

#[inline]
fn alloc_guard(alloc_size: usize) -> Result<(), CollectionAllocErr> {
    if mem::size_of::<usize>() < 8 && alloc_size > core::isize::MAX as usize {
        Err(CapacityOverflow)
    } else {
        Ok(())
    }
}

// One central function responsible for reporting capacity overflows. This'll
// ensure that the code generation related to these panics is minimal as there's
// only one location which panics rather than a bunch throughout the module.
fn capacity_overflow() -> ! {
    panic!("capacity overflow")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allocator_param() {
        use crate::alloc::AllocErr;

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
            unsafe fn alloc(&mut self, layout: Layout) -> Result<NonNull<u8>, AllocErr> {
                let size = layout.size();
                if size > self.fuel {
                    return Err(AllocErr);
                }
                match Global.alloc(layout) {
                    ok @ Ok(_) => { self.fuel -= size; ok }
                    err @ Err(_) => err,
                }
            }
            unsafe fn dealloc(&mut self, ptr: NonNull<u8>, layout: Layout) {
                Global.dealloc(ptr, layout)
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
