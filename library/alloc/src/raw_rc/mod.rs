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

#![allow(dead_code)]

use core::cell::UnsafeCell;

mod rc_layout;

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
