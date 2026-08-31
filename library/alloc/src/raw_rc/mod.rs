//! Base implementation for `rc::{Rc, UniqueRc, Weak}` and `sync::{Arc, UniqueArc, Weak}`.
//!
//! # Allocation Memory Layout
//!
//! The memory layout of a reference-counted allocation is designed so that the memory that stores
//! the reference counts has a fixed offset from the memory that stores the value. In this way,
//! operations that only rely on reference counts can ignore the actual type of the contained value
//! and only care about the address of the contained value, which allows us to share code between
//! reference-counting pointers that have different types of contained values. This can potentially
//! reduce the binary size.
//!
//! Assume the type of the stored value is `T`, the allocation memory layout is designed as follows:
//!
//! - We use a `RefCounts` type to store the reference counts.
//! - The alignment of the allocation is `align_of::<RefCounts>().max(align_of::<T>())`.
//! - The value is stored at offset `size_of::<RefCounts>().next_multiple_of(align_of::<T>())`.
//! - The size of the allocation is
//!   `size_of::<RefCounts>().next_multiple_of(align_of::<T>()) + size_of::<T>()`.
//! - The `RefCounts` object is stored at offset
//!   `size_of::<RefCounts>().next_multiple_of(align_of::<T>()) - size_of::<RefCounts>()`.
//!
//! The following table shows the order and size of each component in a reference-counted allocation
//! of a `T` value:
//!
//! | Component   | Size                                                                                |
//! | ----------- | ----------------------------------------------------------------------------------- |
//! | Padding     | `size_of::<RefCounts>().next_multiple_of(align_of::<T>()) - size_of::<RefCounts>()` |
//! | `RefCounts` | `size_of::<RefCounts>()`                                                            |
//! | `T`         | `size_of::<T>()`                                                                    |
//!
//! This works because:
//!
//! - Both `RefCounts` and the object are stored in the allocation without overlapping.
//! - The `RefCounts` object is stored at offset
//!   `size_of::<RefCounts>().next_multiple_of(align_of::<T>()) - size_of::<RefCounts>()`, which has
//!   a valid alignment for `RefCounts` because:
//!   - If `align_of::<T>() <= align_of::<RefCounts>()`, then the offset is 0, which has a valid
//!     alignment for `RefCounts`.
//!   - If `align_of::<T>() > align_of::<RefCounts>()`, then `align_of::<T>()` is a multiple of
//!     `align_of::<RefCounts>()`. Since `size_of::<RefCounts>()` is also a multiple of
//!     `align_of::<RefCounts>()`, the offset also has a valid alignment for `RefCounts`.
//! - The value is stored at offset `size_of::<RefCounts>().next_multiple_of(align_of::<T>())`,
//!   which trivially satisfies the alignment requirement of `T`.
//! - The distance between the `RefCounts` object and the value is `size_of::<RefCounts>()`, a fixed
//!   value.
//!
//! Thus both the `RefCounts` object and the value have their alignment and size requirements
//! satisfied, and we get a fixed offset between those two objects.
//!
//! # Reference-counting Pointer Design
//!
//! Both strong and weak reference-counting pointers store a pointer to the value object inside the
//! reference-counted allocation, instead of a pointer to the start of the allocation. This is based
//! on the assumption that users access the contained value more frequently than the reference
//! counters. Also, this allows us to enable some possible optimizations such as:
//!
//! - Making reference-counting pointers have ABI-compatible representations as raw pointers so we
//!   can use them directly in FFI interfaces.
//! - Converting `Option<Rc<T>>` to `Option<&T>` without checking for `None` values.
//! - Converting `&[Rc<T>]` to `&[&T]` with zero cost.

use core::cell::UnsafeCell;
use core::mem;
use core::sync::atomic::Atomic;

pub(crate) use crate::raw_rc::raw_rc::RawRc;
pub(crate) use crate::raw_rc::raw_unique_rc::RawUniqueRc;
pub(crate) use crate::raw_rc::raw_weak::RawWeak;

mod raw_rc;
mod raw_unique_rc;
mod raw_weak;
mod rc_alloc;
mod rc_layout;
mod rc_value_pointer;

/// Stores reference counts.
#[cfg_attr(target_pointer_width = "16", repr(C, align(2)))]
#[cfg_attr(target_pointer_width = "32", repr(C, align(4)))]
#[cfg_attr(target_pointer_width = "64", repr(C, align(8)))]
pub(crate) struct RefCounts {
    /// Weak reference count (plus one if there are non-zero strong reference counts).
    pub(crate) weak: UnsafeCell<usize>,
    /// Strong reference count.
    pub(crate) strong: UnsafeCell<usize>,
}

impl RefCounts {
    /// Creates a `RefCounts` with weak count of `1` and strong count of `strong_count`.
    const fn new(strong_count: usize) -> Self {
        Self { weak: UnsafeCell::new(1), strong: UnsafeCell::new(strong_count) }
    }
}

/// The return value type for `RefCounter::make_mut`.
#[cfg(not(no_global_oom_handling))]
pub(crate) enum MakeMutStrategy {
    /// The strong reference count is 1, but weak reference count (including the one shared by all
    /// strong reference count) is more than 1. Before returning, the strong reference count has
    /// been set to zero to prevent new strong pointers from being created through upgrading from
    /// weak pointers.
    Move,
    /// The strong count is more than 1.
    Clone,
}

/// A trait for `rc` and `sync` modules to define their reference-counting behaviors.
///
/// # Safety
///
/// - Each method must be implemented according to its description.
/// - `Self` must have transparent representation over `UnsafeCell<usize>` and every valid
///   `UnsafeCell<usize>` can also be reinterpreted as a valid `Self`.
/// - `Self` must have alignment no greater than `align_of::<Atomic<usize>>()`.
pub(crate) unsafe trait RefCounter: Sized {
    const VERIFY_LAYOUT: () = {
        assert!(size_of::<Self>() == size_of::<UnsafeCell<usize>>());
        assert!(align_of::<Self>() <= align_of::<Atomic<usize>>());
    };

    /// Returns a reference to `Self` from a reference to `UnsafeCell<usize>`.
    ///
    /// # Safety
    ///
    /// - `count` must only be handled by the same `RefCounter` implementation.
    /// - The location of `count` must have enough alignment for storing `Atomic<usize>`.
    unsafe fn from_raw_counter(count: &UnsafeCell<usize>) -> &Self {
        () = Self::VERIFY_LAYOUT;

        // SAFETY: The alignment requirement is guaranteed by both trait implementor and caller.
        // Trait implementor guarantees the alignment of `Self` is not greater than the alignment of
        // `Atomic<usize>`, and caller guarantees that the alignment of `count` is enough for
        // storing `Atomic<usize>`.
        unsafe { mem::transmute(count) }
    }

    /// Increments the reference counter. The process will abort if overflow happens.
    fn increment(&self);

    /// Decrements the reference counter. Returns whether the reference count becomes zero after
    /// decrementing.
    fn decrement(&self) -> bool;

    /// Increments the reference counter if and only if the reference count is non-zero. Returns
    /// whether incrementing is performed.
    fn try_upgrade(&self) -> bool;

    /// Increments the reference counter. If `self` needs to be called with by both
    /// `downgrade_increment_weak` and `is_unique` as the `weak_count` argument concurrently, both
    /// operations will be performed atomically.
    fn downgrade_increment_weak(&self);

    /// Decrements the reference counter if and only if the reference count is 1. Returns true if
    /// decrementing is performed.
    fn try_lock_strong_count(&self) -> bool;

    /// Sets the reference count to 1.
    fn unlock_strong_count(&self);

    /// Returns whether both `strong_count` and `weak_count` are 1. If `weak_count` needs to be
    /// called with by both `downgrade_increment_weak` and `is_unique` concurrently, both operations
    /// will be performed atomically.
    fn is_unique(strong_count: &Self, weak_count: &Self) -> bool;

    /// Determines how to make a mutable reference safely to a reference-counted value.
    ///
    /// - If both strong count and weak count are 1, returns `None`.
    /// - If strong count is 1 and weak count is greater than 1, returns
    ///   `Some(MakeMutStrategy::Move)`.
    /// - If strong count is greater than 1, returns `Some(MakeMutStrategy::Clone)`.
    #[cfg(not(no_global_oom_handling))]
    fn make_mut(strong_count: &Self, weak_count: &Self) -> Option<MakeMutStrategy>;

    /// Returns the weak count of an `RawUniqueRc`, used to determine whether there are any weak
    /// pointers to the same allocation.
    #[cfg(not(no_global_oom_handling))]
    fn unique_rc_weak_count(weak_count: &Self) -> usize;
}
