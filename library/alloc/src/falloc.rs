use core::error::Error;

pub use crate::alloc::{AllocError, Global, GlobalAlloc, Layout, LayoutError};

/// An implementation of `Allocator` can allocate, grow, shrink, and deallocate arbitrary blocks of
/// data described via [`Layout`][].
///
/// `Allocator` is designed to be implemented on ZSTs, references, or smart pointers because having
/// an allocator like `MyAlloc([u8; N])` cannot be moved, without updating the pointers to the
/// allocated memory.
///
/// Unlike [`GlobalAlloc`][], zero-sized allocations are allowed in `Allocator`. If an underlying
/// allocator does not support this (like jemalloc) or return a null pointer (such as
/// `libc::malloc`), this must be caught by the implementation.
///
/// ### Currently allocated memory
///
/// Some of the methods require that a memory block be *currently allocated* via an allocator. This
/// means that:
///
/// * the starting address for that memory block was previously returned by [`allocate`], [`grow`], or
///   [`shrink`], and
///
/// * the memory block has not been subsequently deallocated, where blocks are either deallocated
///   directly by being passed to [`deallocate`] or were changed by being passed to [`grow`] or
///   [`shrink`] that returns `Ok`. If `grow` or `shrink` have returned `Err`, the passed pointer
///   remains valid.
///
/// [`allocate`]: Allocator::allocate
/// [`grow`]: Allocator::grow
/// [`shrink`]: Allocator::shrink
/// [`deallocate`]: Allocator::deallocate
///
/// ### Memory fitting
///
/// Some of the methods require that a layout *fit* a memory block. What it means for a layout to
/// "fit" a memory block means (or equivalently, for a memory block to "fit" a layout) is that the
/// following conditions must hold:
///
/// * The block must be allocated with the same alignment as [`layout.align()`], and
///
/// * The provided [`layout.size()`] must fall in the range `min ..= max`, where:
///   - `min` is the size of the layout most recently used to allocate the block, and
///   - `max` is the latest actual size returned from [`allocate`], [`grow`], or [`shrink`].
///
/// [`layout.align()`]: Layout::align
/// [`layout.size()`]: Layout::size
///
/// # Safety
///
/// * Memory blocks returned from an allocator must point to valid memory and retain their validity
///   until the instance and all of its copies and clones are dropped,
///
/// * copying, cloning, or moving the allocator must not invalidate memory blocks returned from this
///   allocator. A copied or cloned allocator must behave like the same allocator, and
///
/// * any pointer to a memory block which is [*currently allocated*] may be passed to any other
///   method of the allocator.
///
/// [*currently allocated*]: #currently-allocated-memory
#[unstable(feature = "allocator_api", issue = "32838")]
pub unsafe trait Allocator: crate::alloc::Allocator {
    #[must_use] // Doesn't actually work
    type Result<T, E>
    where
        E: Error + IntoLayout;

    #[must_use]
    fn map_result<T, E>(result: Result<T, E>) -> Self::Result<T, E>
    where
        E: Error + IntoLayout;
}

#[cfg(not(no_global_oom_handling))]
use crate::alloc::handle_alloc_error;
#[cfg(not(no_global_oom_handling))]
use crate::collections::{TryReserveError, TryReserveErrorKind};

// One central function responsible for reporting capacity overflows. This'll
// ensure that the code generation related to these panics is minimal as there's
// only one location which panics rather than a bunch throughout the module.
#[cfg(not(no_global_oom_handling))]
pub(crate) fn capacity_overflow() -> ! {
    panic!("capacity overflow");
}

#[unstable(feature = "allocator_api", issue = "32838")]
pub trait IntoLayout {
    #[cfg(not(no_global_oom_handling))]
    fn into_layout(self) -> Layout;
}

#[unstable(feature = "allocator_api", issue = "32838")]
impl IntoLayout for TryReserveError {
    #[cfg(not(no_global_oom_handling))]
    fn into_layout(self) -> Layout {
        match self.kind() {
            TryReserveErrorKind::CapacityOverflow => capacity_overflow(),
            TryReserveErrorKind::AllocError { layout, .. } => layout,
        }
    }
}

#[unstable(feature = "allocator_api", issue = "32838")]
#[cfg(not(no_global_oom_handling))]
unsafe impl<X: crate::alloc::Allocator> Allocator for X {
    type Result<T, E> = T
    where
        E: Error + IntoLayout;

    fn map_result<T, E>(result: Result<T, E>) -> Self::Result<T, E>
    where
        E: Error + IntoLayout,
    {
        result.unwrap_or_else(|error| handle_alloc_error(error.into_layout()))
    }
}
