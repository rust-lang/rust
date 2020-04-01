#![unstable(feature = "raw_vec_internals", reason = "implementation detail", issue = "none")]
#![doc(hidden)]

use core::alloc::MemoryBlock;
use core::cmp;
use core::mem::{self, MaybeUninit};
use core::ops::Drop;
use core::ptr::{NonNull, Unique};
use core::slice;

use crate::alloc::{
    handle_alloc_error, AllocErr,
    AllocInit::{self, *},
    AllocRef, Global, Layout,
    ReallocPlacement::{self, *},
};
use crate::boxed::Box;
use crate::collections::TryReserveError::{self, *};

#[cfg(test)]
mod tests;

/// A low-level utility for more ergonomically allocating, reallocating, and deallocating
/// a buffer of memory on the heap without having to worry about all the corner cases
/// involved. This type is excellent for building your own data structures like Vec and VecDeque.
/// In particular:
///
/// * Produces `Unique::empty()` on zero-sized types.
/// * Produces `Unique::empty()` on zero-length allocations.
/// * Avoids freeing `Unique::empty()`.
/// * Catches all overflows in capacity computations (promotes them to "capacity overflow" panics).
/// * Guards against 32-bit systems allocating more than isize::MAX bytes.
/// * Guards against overflowing your length.
/// * Calls `handle_alloc_error` for fallible allocations.
/// * Contains a `ptr::Unique` and thus endows the user with all related benefits.
/// * Uses the excess returned from the allocator to use the largest available capacity.
///
/// This type does not in anyway inspect the memory that it manages. When dropped it *will*
/// free its memory, but it *won't* try to drop its contents. It is up to the user of `RawVec`
/// to handle the actual things *stored* inside of a `RawVec`.
///
/// Note that the excess of a zero-sized types is always infinite, so `capacity()` always returns
/// `usize::MAX`. This means that you need to be careful when round-tripping this type with a
/// `Box<[T]>`, since `capacity()` won't yield the length.
#[allow(missing_debug_implementations)]
pub struct RawVec<T, A: AllocRef = Global> {
    ptr: Unique<T>,
    cap: usize,
    alloc: A,
}

impl<T> RawVec<T, Global> {
    /// HACK(Centril): This exists because `#[unstable]` `const fn`s needn't conform
    /// to `min_const_fn` and so they cannot be called in `min_const_fn`s either.
    ///
    /// If you change `RawVec<T>::new` or dependencies, please take care to not
    /// introduce anything that would truly violate `min_const_fn`.
    ///
    /// NOTE: We could avoid this hack and check conformance with some
    /// `#[rustc_force_min_const_fn]` attribute which requires conformance
    /// with `min_const_fn` but does not necessarily allow calling it in
    /// `stable(...) const fn` / user code not enabling `foo` when
    /// `#[rustc_const_unstable(feature = "foo", ..)]` is present.
    pub const NEW: Self = Self::new();

    /// Creates the biggest possible `RawVec` (on the system heap)
    /// without allocating. If `T` has positive size, then this makes a
    /// `RawVec` with capacity `0`. If `T` is zero-sized, then it makes a
    /// `RawVec` with capacity `usize::MAX`. Useful for implementing
    /// delayed allocation.
    pub const fn new() -> Self {
        Self::new_in(Global)
    }

    /// Creates a `RawVec` (on the system heap) with exactly the
    /// capacity and alignment requirements for a `[T; capacity]`. This is
    /// equivalent to calling `RawVec::new` when `capacity` is `0` or `T` is
    /// zero-sized. Note that if `T` is zero-sized this means you will
    /// *not* get a `RawVec` with the requested capacity.
    ///
    /// # Panics
    ///
    /// * Panics if the requested capacity exceeds `usize::MAX` bytes.
    /// * Panics on 32-bit platforms if the requested capacity exceeds
    ///   `isize::MAX` bytes.
    ///
    /// # Aborts
    ///
    /// Aborts on OOM.
    #[inline]
    pub fn with_capacity(capacity: usize) -> Self {
        Self::with_capacity_in(capacity, Global)
    }

    /// Like `with_capacity`, but guarantees the buffer is zeroed.
    #[inline]
    pub fn with_capacity_zeroed(capacity: usize) -> Self {
        Self::with_capacity_zeroed_in(capacity, Global)
    }

    /// Reconstitutes a `RawVec` from a pointer and capacity.
    ///
    /// # Safety
    ///
    /// The `ptr` must be allocated (on the system heap), and with the given `capacity`.
    /// The `capacity` cannot exceed `isize::MAX` for sized types. (only a concern on 32-bit
    /// systems). ZST vectors may have a capacity up to `usize::MAX`.
    /// If the `ptr` and `capacity` come from a `RawVec`, then this is guaranteed.
    #[inline]
    pub unsafe fn from_raw_parts(ptr: *mut T, capacity: usize) -> Self {
        Self::from_raw_parts_in(ptr, capacity, Global)
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

impl<T, A: AllocRef> RawVec<T, A> {
    /// Like `new`, but parameterized over the choice of allocator for
    /// the returned `RawVec`.
    pub const fn new_in(alloc: A) -> Self {
        // `cap: 0` means "unallocated". zero-sized types are ignored.
        Self { ptr: Unique::empty(), cap: 0, alloc }
    }

    /// Like `with_capacity`, but parameterized over the choice of
    /// allocator for the returned `RawVec`.
    #[inline]
    pub fn with_capacity_in(capacity: usize, alloc: A) -> Self {
        Self::allocate_in(capacity, Uninitialized, alloc)
    }

    /// Like `with_capacity_zeroed`, but parameterized over the choice
    /// of allocator for the returned `RawVec`.
    #[inline]
    pub fn with_capacity_zeroed_in(capacity: usize, alloc: A) -> Self {
        Self::allocate_in(capacity, Zeroed, alloc)
    }

    fn allocate_in(capacity: usize, init: AllocInit, mut alloc: A) -> Self {
        if mem::size_of::<T>() == 0 {
            Self::new_in(alloc)
        } else {
            let layout = Layout::array::<T>(capacity).unwrap_or_else(|_| capacity_overflow());
            alloc_guard(layout.size()).unwrap_or_else(|_| capacity_overflow());

            let memory = alloc.alloc(layout, init).unwrap_or_else(|_| handle_alloc_error(layout));
            Self {
                ptr: memory.ptr.cast().into(),
                cap: Self::capacity_from_bytes(memory.size),
                alloc,
            }
        }
    }

    /// Reconstitutes a `RawVec` from a pointer, capacity, and allocator.
    ///
    /// # Safety
    ///
    /// The `ptr` must be allocated (via the given allocator `a`), and with the given `capacity`.
    /// The `capacity` cannot exceed `isize::MAX` for sized types. (only a concern on 32-bit
    /// systems). ZST vectors may have a capacity up to `usize::MAX`.
    /// If the `ptr` and `capacity` come from a `RawVec` created via `a`, then this is guaranteed.
    #[inline]
    pub unsafe fn from_raw_parts_in(ptr: *mut T, capacity: usize, a: A) -> Self {
        Self { ptr: Unique::new_unchecked(ptr), cap: capacity, alloc: a }
    }

    /// Gets a raw pointer to the start of the allocation. Note that this is
    /// `Unique::empty()` if `capacity == 0` or `T` is zero-sized. In the former case, you must
    /// be careful.
    pub fn ptr(&self) -> *mut T {
        self.ptr.as_ptr()
    }

    /// Gets the capacity of the allocation.
    ///
    /// This will always be `usize::MAX` if `T` is zero-sized.
    #[inline(always)]
    pub fn capacity(&self) -> usize {
        if mem::size_of::<T>() == 0 { usize::MAX } else { self.cap }
    }

    /// Returns a shared reference to the allocator backing this `RawVec`.
    pub fn alloc(&self) -> &A {
        &self.alloc
    }

    /// Returns a mutable reference to the allocator backing this `RawVec`.
    pub fn alloc_mut(&mut self) -> &mut A {
        &mut self.alloc
    }

    fn current_memory(&self) -> Option<(NonNull<u8>, Layout)> {
        if mem::size_of::<T>() == 0 || self.cap == 0 {
            None
        } else {
            // We have an allocated chunk of memory, so we can bypass runtime
            // checks to get our current layout.
            unsafe {
                let align = mem::align_of::<T>();
                let size = mem::size_of::<T>() * self.cap;
                let layout = Layout::from_size_align_unchecked(size, align);
                Some((self.ptr.cast().into(), layout))
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
    /// manually check if your `len == capacity`.
    ///
    /// # Panics
    ///
    /// * Panics if `T` is zero-sized on the assumption that you managed to exhaust
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
    ///         if self.len == self.buf.capacity() { self.buf.double(); }
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
        match self.grow(Double, MayMove, Uninitialized) {
            Err(CapacityOverflow) => capacity_overflow(),
            Err(AllocError { layout, .. }) => handle_alloc_error(layout),
            Ok(()) => { /* yay */ }
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
    /// * Panics if `T` is zero-sized on the assumption that you managed to exhaust
    ///   all `usize::MAX` slots in your imaginary buffer.
    /// * Panics on 32-bit platforms if the requested capacity exceeds
    ///   `isize::MAX` bytes.
    #[inline(never)]
    #[cold]
    pub fn double_in_place(&mut self) -> bool {
        self.grow(Double, InPlace, Uninitialized).is_ok()
    }

    /// Ensures that the buffer contains at least enough space to hold
    /// `used_capacity + needed_extra_capacity` elements. If it doesn't already have
    /// enough capacity, will reallocate enough space plus comfortable slack
    /// space to get amortized `O(1)` behavior. Will limit this behavior
    /// if it would needlessly cause itself to panic.
    ///
    /// If `used_capacity` exceeds `self.capacity()`, this may fail to actually allocate
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
    /// Aborts on OOM.
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
    pub fn reserve(&mut self, used_capacity: usize, needed_extra_capacity: usize) {
        match self.try_reserve(used_capacity, needed_extra_capacity) {
            Err(CapacityOverflow) => capacity_overflow(),
            Err(AllocError { layout, .. }) => handle_alloc_error(layout),
            Ok(()) => { /* yay */ }
        }
    }

    /// The same as `reserve`, but returns on errors instead of panicking or aborting.
    pub fn try_reserve(
        &mut self,
        used_capacity: usize,
        needed_extra_capacity: usize,
    ) -> Result<(), TryReserveError> {
        if self.needs_to_grow(used_capacity, needed_extra_capacity) {
            self.grow(Amortized { used_capacity, needed_extra_capacity }, MayMove, Uninitialized)
        } else {
            Ok(())
        }
    }

    /// Attempts to ensure that the buffer contains at least enough space to hold
    /// `used_capacity + needed_extra_capacity` elements. If it doesn't already have
    /// enough capacity, will reallocate in place enough space plus comfortable slack
    /// space to get amortized `O(1)` behavior. Will limit this behaviour
    /// if it would needlessly cause itself to panic.
    ///
    /// If `used_capacity` exceeds `self.capacity()`, this may fail to actually allocate
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
    pub fn reserve_in_place(&mut self, used_capacity: usize, needed_extra_capacity: usize) -> bool {
        // This is more readable than putting this in one line:
        // `!self.needs_to_grow(...) || self.grow(...).is_ok()`
        if self.needs_to_grow(used_capacity, needed_extra_capacity) {
            self.grow(Amortized { used_capacity, needed_extra_capacity }, InPlace, Uninitialized)
                .is_ok()
        } else {
            true
        }
    }

    /// Ensures that the buffer contains at least enough space to hold
    /// `used_capacity + needed_extra_capacity` elements. If it doesn't already,
    /// will reallocate the minimum possible amount of memory necessary.
    /// Generally this will be exactly the amount of memory necessary,
    /// but in principle the allocator is free to give back more than
    /// we asked for.
    ///
    /// If `used_capacity` exceeds `self.capacity()`, this may fail to actually allocate
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
    /// Aborts on OOM.
    pub fn reserve_exact(&mut self, used_capacity: usize, needed_extra_capacity: usize) {
        match self.try_reserve_exact(used_capacity, needed_extra_capacity) {
            Err(CapacityOverflow) => capacity_overflow(),
            Err(AllocError { layout, .. }) => handle_alloc_error(layout),
            Ok(()) => { /* yay */ }
        }
    }

    /// The same as `reserve_exact`, but returns on errors instead of panicking or aborting.
    pub fn try_reserve_exact(
        &mut self,
        used_capacity: usize,
        needed_extra_capacity: usize,
    ) -> Result<(), TryReserveError> {
        if self.needs_to_grow(used_capacity, needed_extra_capacity) {
            self.grow(Exact { used_capacity, needed_extra_capacity }, MayMove, Uninitialized)
        } else {
            Ok(())
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
        match self.shrink(amount, MayMove) {
            Err(CapacityOverflow) => capacity_overflow(),
            Err(AllocError { layout, .. }) => handle_alloc_error(layout),
            Ok(()) => { /* yay */ }
        }
    }
}

#[derive(Copy, Clone)]
enum Strategy {
    Double,
    Amortized { used_capacity: usize, needed_extra_capacity: usize },
    Exact { used_capacity: usize, needed_extra_capacity: usize },
}
use Strategy::*;

impl<T, A: AllocRef> RawVec<T, A> {
    /// Returns if the buffer needs to grow to fulfill the needed extra capacity.
    /// Mainly used to make inlining reserve-calls possible without inlining `grow`.
    fn needs_to_grow(&self, used_capacity: usize, needed_extra_capacity: usize) -> bool {
        needed_extra_capacity > self.capacity().wrapping_sub(used_capacity)
    }

    fn capacity_from_bytes(excess: usize) -> usize {
        debug_assert_ne!(mem::size_of::<T>(), 0);
        excess / mem::size_of::<T>()
    }

    fn set_memory(&mut self, memory: MemoryBlock) {
        self.ptr = memory.ptr.cast().into();
        self.cap = Self::capacity_from_bytes(memory.size);
    }

    /// Single method to handle all possibilities of growing the buffer.
    fn grow(
        &mut self,
        strategy: Strategy,
        placement: ReallocPlacement,
        init: AllocInit,
    ) -> Result<(), TryReserveError> {
        let elem_size = mem::size_of::<T>();
        if elem_size == 0 {
            // Since we return a capacity of `usize::MAX` when `elem_size` is
            // 0, getting to here necessarily means the `RawVec` is overfull.
            return Err(CapacityOverflow);
        }
        let new_layout = match strategy {
            Double => unsafe {
                // Since we guarantee that we never allocate more than `isize::MAX` bytes,
                // `elem_size * self.cap <= isize::MAX` as a precondition, so this can't overflow.
                // Additionally the alignment will never be too large as to "not be satisfiable",
                // so `Layout::from_size_align` will always return `Some`.
                //
                // TL;DR, we bypass runtime checks due to dynamic assertions in this module,
                // allowing us to use `from_size_align_unchecked`.
                let cap = if self.cap == 0 {
                    // Skip to 4 because tiny `Vec`'s are dumb; but not if that would cause overflow.
                    if elem_size > usize::MAX / 8 { 1 } else { 4 }
                } else {
                    self.cap * 2
                };
                Layout::from_size_align_unchecked(cap * elem_size, mem::align_of::<T>())
            },
            Amortized { used_capacity, needed_extra_capacity } => {
                // Nothing we can really do about these checks, sadly.
                let required_cap =
                    used_capacity.checked_add(needed_extra_capacity).ok_or(CapacityOverflow)?;
                // Cannot overflow, because `cap <= isize::MAX`, and type of `cap` is `usize`.
                let double_cap = self.cap * 2;
                // `double_cap` guarantees exponential growth.
                let cap = cmp::max(double_cap, required_cap);
                Layout::array::<T>(cap).map_err(|_| CapacityOverflow)?
            }
            Exact { used_capacity, needed_extra_capacity } => {
                let cap =
                    used_capacity.checked_add(needed_extra_capacity).ok_or(CapacityOverflow)?;
                Layout::array::<T>(cap).map_err(|_| CapacityOverflow)?
            }
        };
        alloc_guard(new_layout.size())?;

        let memory = if let Some((ptr, old_layout)) = self.current_memory() {
            debug_assert_eq!(old_layout.align(), new_layout.align());
            unsafe {
                self.alloc
                    .grow(ptr, old_layout, new_layout.size(), placement, init)
                    .map_err(|_| AllocError { layout: new_layout, non_exhaustive: () })?
            }
        } else {
            match placement {
                MayMove => self.alloc.alloc(new_layout, init),
                InPlace => Err(AllocErr),
            }
            .map_err(|_| AllocError { layout: new_layout, non_exhaustive: () })?
        };
        self.set_memory(memory);
        Ok(())
    }

    fn shrink(
        &mut self,
        amount: usize,
        placement: ReallocPlacement,
    ) -> Result<(), TryReserveError> {
        assert!(amount <= self.capacity(), "Tried to shrink to a larger capacity");

        let (ptr, layout) = if let Some(mem) = self.current_memory() { mem } else { return Ok(()) };
        let new_size = amount * mem::size_of::<T>();

        let memory = unsafe {
            self.alloc.shrink(ptr, layout, new_size, placement).map_err(|_| {
                TryReserveError::AllocError {
                    layout: Layout::from_size_align_unchecked(new_size, layout.align()),
                    non_exhaustive: (),
                }
            })?
        };
        self.set_memory(memory);
        Ok(())
    }
}

impl<T> RawVec<T, Global> {
    /// Converts the entire buffer into `Box<[MaybeUninit<T>]>` with the specified `len`.
    ///
    /// Note that this will correctly reconstitute any `cap` changes
    /// that may have been performed. (See description of type for details.)
    ///
    /// # Safety
    ///
    /// `shrink_to_fit(len)` must be called immediately prior to calling this function. This
    /// implies, that `len` must be smaller than or equal to `self.capacity()`.
    pub unsafe fn into_box(self, len: usize) -> Box<[MaybeUninit<T>]> {
        debug_assert!(
            len <= self.capacity(),
            "`len` must be smaller than or equal to `self.capacity()`"
        );

        // NOTE: not calling `capacity()` here; actually using the real `cap` field!
        let slice = slice::from_raw_parts_mut(self.ptr() as *mut MaybeUninit<T>, len);
        let output = Box::from_raw(slice);
        mem::forget(self);
        output
    }
}

unsafe impl<#[may_dangle] T, A: AllocRef> Drop for RawVec<T, A> {
    /// Frees the memory owned by the `RawVec` *without* trying to drop its contents.
    fn drop(&mut self) {
        if let Some((ptr, layout)) = self.current_memory() {
            unsafe { self.alloc.dealloc(ptr, layout) }
        }
    }
}

// We need to guarantee the following:
// * We don't ever allocate `> isize::MAX` byte-size objects.
// * We don't overflow `usize::MAX` and actually allocate too little.
//
// On 64-bit we just need to check for overflow since trying to allocate
// `> isize::MAX` bytes will surely fail. On 32-bit and 16-bit we need to add
// an extra guard for this in case we're running on a platform which can use
// all 4GB in user-space, e.g., PAE or x32.

#[inline]
fn alloc_guard(alloc_size: usize) -> Result<(), TryReserveError> {
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
    panic!("capacity overflow");
}
