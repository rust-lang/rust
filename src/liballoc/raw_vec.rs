#![unstable(feature = "raw_vec_internals", reason = "implementation detail", issue = "none")]
#![doc(hidden)]

use core::cmp;
use core::mem;
use core::ops::Drop;
use core::ptr::Unique;
use core::slice;

use crate::alloc::{handle_alloc_error, AllocErr, AllocRef, Global, Layout};
use crate::boxed::Box;
use crate::collections::TryReserveError::{self, *};

#[cfg(test)]
mod tests;

/// A low-level utility for more ergonomically allocating, reallocating, and deallocating
/// a buffer of memory on the heap without having to worry about all the corner cases
/// involved. This type is excellent for building your own data structures like Vec and VecDeque.
/// In particular:
///
/// * Catches all overflows in capacity computations (promotes them to "capacity overflow" panics).
/// * Guards against 32-bit systems allocating more than isize::MAX bytes.
/// * Guards against overflowing your length.
/// * Calls `handle_alloc_error` for fallible allocations.
/// * Contains a `ptr::Unique` and thus endows the user with all related benefits.
///
/// This type does not in anyway inspect the memory that it manages. When dropped it *will*
/// free its memory, but it *won't* try to drop its contents. It is up to the user of `RawVec`
/// to handle the actual things *stored* inside of a `RawVec`.
///
/// Note that a `RawVec` always returns a capacity of `usize::MAX` for zero-sized types. Beside
/// this, zero-sized types are handled like non-zero-sized types.
#[allow(missing_debug_implementations)]
pub struct RawVec<T, A: AllocRef = Global> {
    ptr: Unique<T>,
    cap: usize,
    a: A,
}

impl<T, A: AllocRef> RawVec<T, A> {
    /// Like `new`, but parameterized over the choice of allocator for
    /// the returned `RawVec`.
    pub const fn new_in(a: A) -> Self {
        // `cap: 0` means "unallocated". zero-sized allocations are handled by `AllocRef`
        Self { ptr: Unique::empty(), cap: 0, a }
    }

    /// Like `with_capacity`, but parameterized over the choice of
    /// allocator for the returned `RawVec`.
    #[inline]
    pub fn with_capacity_in(capacity: usize, a: A) -> Self {
        Self::allocate_in(capacity, AllocInit::Unspecified, a)
    }

    /// Like `with_capacity_zeroed`, but parameterized over the choice
    /// of allocator for the returned `RawVec`.
    #[inline]
    pub fn with_capacity_zeroed_in(capacity: usize, a: A) -> Self {
        Self::allocate_in(capacity, AllocInit::Zero, a)
    }

    fn allocate_in(capacity: usize, init: AllocInit, mut a: A) -> Self {
        let layout = Layout::array::<T>(capacity).unwrap_or_else(|_| capacity_overflow());
        alloc_guard(layout.size()).unwrap_or_else(|_| capacity_overflow());

        let allocation = match init {
            AllocInit::Unspecified => a.alloc(layout),
            AllocInit::Zero => a.alloc_zeroed(layout),
        };
        let (ptr, alloc_size) = allocation.unwrap_or_else(|_| handle_alloc_error(layout));

        let ptr = ptr.cast().as_ptr();
        let elem_size = mem::size_of::<T>();
        unsafe {
            if elem_size == 0 {
                Self::from_raw_parts_in(ptr, capacity, a)
            } else {
                Self::from_raw_parts_in(ptr, alloc_size / elem_size, a)
            }
        }
    }
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
}

impl<T, A: AllocRef> RawVec<T, A> {
    /// Reconstitutes a `RawVec` from a pointer, capacity, and allocator.
    ///
    /// # Undefined Behavior
    ///
    /// The `ptr` must be allocated (via the given allocator `a`), and with the given `capacity`.
    /// The `capacity` cannot exceed `isize::MAX` (only a concern on 32-bit systems).
    /// If the `ptr` and `capacity` come from a `RawVec` created via `a`, then this is guaranteed.
    #[inline]
    pub unsafe fn from_raw_parts_in(ptr: *mut T, capacity: usize, a: A) -> Self {
        Self { ptr: Unique::new_unchecked(ptr), cap: capacity, a }
    }
}

impl<T> RawVec<T, Global> {
    /// Reconstitutes a `RawVec` from a pointer and capacity.
    ///
    /// # Undefined Behavior
    ///
    /// The `ptr` must be allocated (on the system heap), and with the given `capacity`.
    /// The `capacity` cannot exceed `isize::MAX` (only a concern on 32-bit systems).
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
        &self.a
    }

    /// Returns a mutable reference to the allocator backing this `RawVec`.
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
    /// manually check if your `len == capacity`.
    ///
    /// # Panics
    ///
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
        match self.grow(Double, AllocPlacement::Unspecified, AllocInit::Unspecified) {
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
    /// * Panics on 32-bit platforms if the requested capacity exceeds
    ///   `isize::MAX` bytes.
    #[inline(never)]
    #[cold]
    pub fn double_in_place(&mut self) -> bool {
        self.grow(Double, AllocPlacement::InPlace, AllocInit::Unspecified).is_ok()
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
            self.grow(
                Amortized { used_capacity, needed_extra_capacity },
                AllocPlacement::Unspecified,
                AllocInit::Unspecified,
            )
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
        // This is more readable as putting this in one line:
        // `!self.needs_to_grow(...) || self.grow(...).is_ok()`
        if self.needs_to_grow(used_capacity, needed_extra_capacity) {
            self.grow(
                Amortized { used_capacity, needed_extra_capacity },
                AllocPlacement::InPlace,
                AllocInit::Unspecified,
            )
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
            self.grow(
                Exact { used_capacity, needed_extra_capacity },
                AllocPlacement::Unspecified,
                AllocInit::Unspecified,
            )
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
        match self.shrink(amount, AllocPlacement::Unspecified) {
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

enum AllocInit {
    Unspecified,
    Zero,
}

enum AllocPlacement {
    Unspecified,
    InPlace,
}

impl<T, A: AllocRef> RawVec<T, A> {
    /// Returns if the buffer needs to grow to fulfill the needed extra capacity.
    /// Mainly used to make inlining reserve-calls possible without inlining `grow`.
    fn needs_to_grow(&self, used_capacity: usize, needed_extra_capacity: usize) -> bool {
        needed_extra_capacity > self.capacity().wrapping_sub(used_capacity)
    }

    /// Single method to handle all possibilities of growing the buffer.
    fn grow(
        &mut self,
        strategy: Strategy,
        placement: AllocPlacement,
        init: AllocInit,
    ) -> Result<(), TryReserveError> {
        let elem_size = mem::size_of::<T>();
        let (new_layout, new_cap) = match strategy {
            Double => unsafe {
                if elem_size == 0 {
                    // Since we return a capacity of `usize::MAX` when `elem_size` is
                    // 0, getting to here necessarily means the `RawVec` is overfull.
                    return Err(CapacityOverflow);
                }
                // Since we guarantee that we never allocate more than `isize::MAX` bytes,
                // `elem_size * self.cap <= isize::MAX` as a precondition, so this can't overflow.
                // Additionally the alignment will never be too large as to "not be satisfiable",
                // so `Layout::from_size_align` will always return `Some`.
                //
                // TL;DR, we bypass runtime checks due to dynamic assertions in this module,
                // allowing us to use `from_size_align_unchecked`.
                let cap = if self.cap == 0 {
                    if elem_size > usize::MAX / 8 { 1 } else { 4 }
                } else {
                    self.cap * 2
                };
                let layout =
                    Layout::from_size_align_unchecked(cap * elem_size, mem::align_of::<T>());
                (layout, cap)
            },
            Amortized { used_capacity, needed_extra_capacity } => {
                // Nothing we can really do about these checks, sadly.
                let required_cap =
                    used_capacity.checked_add(needed_extra_capacity).ok_or(CapacityOverflow)?;
                // Cannot overflow, because `cap <= isize::MAX`, and type of `cap` is `usize`.
                let double_cap = self.cap * 2;
                // `double_cap` guarantees exponential growth.
                let cap = cmp::max(double_cap, required_cap);
                let layout = Layout::array::<T>(cap).map_err(|_| CapacityOverflow)?;
                (layout, cap)
            }
            Exact { used_capacity, needed_extra_capacity } => {
                let cap =
                    used_capacity.checked_add(needed_extra_capacity).ok_or(CapacityOverflow)?;
                let layout = Layout::array::<T>(cap).map_err(|_| CapacityOverflow)?;
                (layout, cap)
            }
        };

        let allocation = if let Some(old_layout) = self.current_layout() {
            debug_assert!(old_layout.align() == new_layout.align());
            debug_assert!(old_layout.size() <= new_layout.size());
            let ptr = self.ptr.cast().into();
            unsafe {
                match (placement, init) {
                    (AllocPlacement::Unspecified, AllocInit::Unspecified) => {
                        self.a.realloc(ptr, old_layout, new_layout.size())
                    }
                    (AllocPlacement::Unspecified, AllocInit::Zero) => {
                        self.a.realloc_zeroed(ptr, old_layout, new_layout.size())
                    }
                    (AllocPlacement::InPlace, AllocInit::Unspecified) => self
                        .a
                        .grow_in_place(ptr, old_layout, new_layout.size())
                        .map(|size| (ptr, size))
                        .map_err(|_| AllocErr),
                    (AllocPlacement::InPlace, AllocInit::Zero) => self
                        .a
                        .grow_in_place_zeroed(ptr, old_layout, new_layout.size())
                        .map(|size| (ptr, size))
                        .map_err(|_| AllocErr),
                }
            }
        } else {
            match (placement, init) {
                (AllocPlacement::Unspecified, AllocInit::Unspecified) => self.a.alloc(new_layout),
                (AllocPlacement::Unspecified, AllocInit::Zero) => self.a.alloc_zeroed(new_layout),
                (AllocPlacement::InPlace, _) => Err(AllocErr),
            }
        };
        allocation
            .map(|(ptr, alloc_size)| {
                self.ptr = ptr.cast().into();
                if elem_size == 0 {
                    self.cap = new_cap;
                } else {
                    self.cap = alloc_size / elem_size;
                }
            })
            .map_err(|_| TryReserveError::AllocError { layout: new_layout, non_exhaustive: () })
    }

    fn shrink(&mut self, amount: usize, placement: AllocPlacement) -> Result<(), TryReserveError> {
        assert!(amount <= self.cap, "Tried to shrink to a larger capacity");

        let elem_size = mem::size_of::<T>();
        let old_layout =
            if let Some(layout) = self.current_layout() { layout } else { return Ok(()) };
        let old_ptr = self.ptr.cast().into();
        let new_size = amount * elem_size;

        let allocation = unsafe {
            match (amount, placement) {
                (0, AllocPlacement::Unspecified) => {
                    self.dealloc_buffer();
                    Ok((old_layout.dangling(), 0))
                }
                (_, AllocPlacement::Unspecified) => self.a.realloc(old_ptr, old_layout, new_size),
                (_, AllocPlacement::InPlace) => self
                    .a
                    .shrink_in_place(old_ptr, old_layout, new_size)
                    .map(|size| (old_ptr, size))
                    .map_err(|_| AllocErr),
            }
        };

        allocation
            .map(|(ptr, alloc_size)| {
                self.ptr = ptr.cast().into();
                if elem_size == 0 {
                    self.cap = amount;
                } else {
                    self.cap = alloc_size / elem_size;
                }
            })
            .map_err(|_| TryReserveError::AllocError {
                layout: unsafe { Layout::from_size_align_unchecked(new_size, old_layout.align()) },
                non_exhaustive: (),
            })
    }
}

impl<T> RawVec<T, Global> {
    /// Converts the entire buffer into `Box<[T]>`.
    ///
    /// Note that this will correctly reconstitute any `cap` changes
    /// that may have been performed. (See description of type for details.)
    ///
    /// # Undefined Behavior
    ///
    /// All elements of `RawVec<T, Global>` must be initialized. Notice that
    /// the rules around uninitialized boxed values are not finalized yet,
    /// but until they are, it is advisable to avoid them.
    pub unsafe fn into_box(self) -> Box<[T]> {
        // NOTE: not calling `capacity()` here; actually using the real `cap` field!
        let slice = slice::from_raw_parts_mut(self.ptr(), self.cap);
        let output: Box<[T]> = Box::from_raw(slice);
        mem::forget(self);
        output
    }
}

impl<T, A: AllocRef> RawVec<T, A> {
    /// Frees the memory owned by the `RawVec` *without* trying to drop its contents.
    pub unsafe fn dealloc_buffer(&mut self) {
        if let Some(layout) = self.current_layout() {
            self.a.dealloc(self.ptr.cast().into(), layout);
        }
    }
}

unsafe impl<#[may_dangle] T, A: AllocRef> Drop for RawVec<T, A> {
    /// Frees the memory owned by the `RawVec` *without* trying to drop its contents.
    fn drop(&mut self) {
        unsafe { self.dealloc_buffer() }
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
