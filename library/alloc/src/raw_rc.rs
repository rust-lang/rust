//! Base implementation for `rc::{Rc, Weak}` and `sync::{Arc, Weak}`.
//!
//! # Allocation Memory Layout
//!
//! The memory layout of a reference-counted allocation is designed so that the memory that stores
//! the reference counts has a fixed offset to the memory that stores the value. In this way,
//! operations that only rely on reference counts can ignore the actual type of the contained value
//! and only care about the address of the contained value, which allows us to share code between
//! reference-counting pointers that have different types of contained values. This can potentially
//! reduce the binary size.
//!
//! Assume the type of the stored value is `T`, the allocation memory layout is designed as follows:
//!
//! - We use a `RefCounts` type that has `size_of::<RefCounts>().is_power_of_two()` to store the
//!   reference counts.
//! - The size of the allocation is `size_of::<RefCounts>().max(align_of::<T>()) + size_of::<T>()`.
//! - The alignment of the allocation is `align_of::<RefCounts>().max(align_of::<T>())`.
//! - The `RefCounts` object is stored at offset
//!   `align_of::<T>().saturating_sub(size_of::<RefCounts>())`.
//! - The value is stored at offset `size_of::<RefCounts>().max(align_of::<T>())`.
//!
//! This works because:
//!
//! - Offset `size_of::<RefCounts>().max(align_of::<T>())` where the value is stored has the
//!   alignment of `align_of::<RefCounts>().max(align_of::<T>())` because `size_of::<RefCounts>()`
//!   must be a multiple of `align_of::<RefCounts>()`.
//! - Offset `align_of::<T>().saturating_sub(size_of::<RefCounts>())` where the `RefCounts` object
//!   is stored equals `size_of::<RefCounts>().max(align_of::<T>()) - size_of::<RefCounts>()`, so
//!   it has the alignment of `align_of::<RefCounts>()`, which is a valid alignment for `RefCounts`.
//! - The distance between the `RefCounts` object and the value is:
//!   `size_of::<RefCounts>().max(align_of::<T>()) - align_of::<T>().saturating_sub(size_of::<RefCounts>())`,
//!   which equals `size_of::<RefCounts>()`, which is a fixed offset and ensures that the
//!   `RefCounts` object and the value object do not overlap.
//! - The size of the allocation is `size_of::<RefCounts>().max(align_of::<T>()) + size_of::<T>()`,
//!   and the value is stored at `size_of::<RefCounts>().max(align_of::<T>())`, which leaves exactly
//!   `size_of::<T>()` bytes for storing the value object.
//!
//! So both the `RefCounts` object and the value object have their alignment and size requirements.
//! And we get a fixed offset between those two objects.
//!
//! # Reference-counting Pointer Design
//!
//! Both strong and weak reference-counting pointers store a pointer that points to the value
//! object in a reference-counted allocation instead of the beginning of the allocation. This is
//! based on the assumption that users access the contained value more frequently than the reference
//! counters. Also, this possibly allows us to enable some future optimizations like:
//!
//! - Making reference-counting pointers have ABI-compatible representation as raw pointers so we
//!   can use them directly in FFI interfaces.
//! - Converting `Option<Rc<T>>` to `Option<&T>` with a memory copy operation.
//! - Converting `&[Rc<T>]` to `&[&T]` with zero cost.

use core::alloc::{AllocError, Allocator, Layout, LayoutError};
use core::any::Any;
use core::cell::UnsafeCell;
#[cfg(not(no_global_oom_handling))]
use core::clone::CloneToUninit;
use core::error::{Error, Request};
use core::fmt::{self, Debug, Display, Formatter, Pointer};
use core::hash::{Hash, Hasher};
#[cfg(not(no_global_oom_handling))]
use core::iter::TrustedLen;
use core::marker::{PhantomData, Unsize};
#[cfg(not(no_global_oom_handling))]
use core::mem::ManuallyDrop;
use core::mem::{self, MaybeUninit, SizedTypeProperties};
use core::num::NonZeroUsize;
use core::ops::{CoerceUnsized, Deref, DispatchFromDyn};
use core::pin::PinCoerceUnsized;
use core::ptr::{self, NonNull};
use core::str;

#[cfg(not(no_global_oom_handling))]
use crate::alloc;
use crate::alloc::Global;
#[cfg(not(no_global_oom_handling))]
use crate::boxed::Box;
#[cfg(not(no_global_oom_handling))]
use crate::string::String;
#[cfg(not(no_global_oom_handling))]
use crate::vec::Vec;

/// The return value type for `RcOps::make_mut`.
#[cfg(not(no_global_oom_handling))]
pub(crate) enum MakeMutStrategy {
    /// This `RawRc` is the only strong pointer that references the value, but there are weak
    /// pointers also referencing the value. Before returning, the strong reference count has been
    /// set to zero to prevent new strong pointers from being created through upgrading from weak
    /// pointers.
    Move,
    /// There is more than one strong pointer that references the value.
    Clone,
}

/// A trait for `rc` and `sync` modules to define their own implementation of reference-counting
/// behaviors.
///
/// # Safety
///
/// Implementors should implement each method according to its description.
pub(crate) unsafe trait RcOps: Sized {
    /// Increment a reference counter managed by `RawRc` and `RawWeak`. Currently, both strong and
    /// weak reference counters are incremented by this method.
    ///
    /// # Safety
    ///
    /// - `count` should only be handled by the same `RcOps` implementation.
    /// - The value of `count` should be non-zero.
    unsafe fn increment_ref_count(count: &UnsafeCell<usize>);

    /// Decrement a reference counter managed by `RawRc` and `RawWeak`. Currently, both strong and
    /// weak reference counters are decremented by this method. Returns whether the reference count
    /// becomes zero after decrementing.
    ///
    /// # Safety
    ///
    /// - `count` should only be handled by the same `RcOps` implementation.
    /// - The value of `count` should be non-zero.
    unsafe fn decrement_ref_count(count: &UnsafeCell<usize>) -> bool;

    /// Increment `strong_count` if and only if `strong_count` is non-zero. Returns whether
    /// incrementing is performed.
    ///
    /// # Safety
    ///
    /// - `strong_count` should be provided by a `RawWeak` object.
    /// - `strong_count` should only be handled by the same `RcOps` implementation.
    unsafe fn upgrade(strong_count: &UnsafeCell<usize>) -> bool;

    /// Increment `weak_count`. This is required instead of `increment_ref_count` because `Arc`
    /// requires additional synchronization with `is_unique`.
    ///
    /// # Safety
    ///
    /// - `weak_count` should be provided by a `RawRc` object.
    /// - `weak_count` should only be handled by the same `RcOps` implementation.
    /// - Caller should provide a `weak_count` value from a `RawRc` object.
    unsafe fn downgrade(weak_count: &UnsafeCell<usize>);

    /// Decrement `strong_count` if and only if `strong_count` is 1. Returns true if decrementing
    /// is performed.
    ///
    /// # Safety
    ///
    /// - `strong_count` should be provided by a `RawRc` object.
    /// - `strong_count` should only be handled by the same `RcOps` implementation.
    unsafe fn lock_strong_count(strong_count: &UnsafeCell<usize>) -> bool;

    /// Set `strong_count` to 1.
    ///
    /// # Safety
    ///
    /// - `strong_count` should be provided by a `RawUniqueRc` object.
    /// - `strong_count` should only be handled by the same `RcOps` implementation.
    unsafe fn unlock_strong_count(strong_count: &UnsafeCell<usize>);

    /// Returns whether both `strong_count` are 1 and `weak_count` is 1. Used by `RawRc::get_mut`.
    ///
    /// # Safety
    ///
    /// - `ref_counts` should be provided by a `RawRc` object.
    /// - `ref_counts` should only be handled by the same `RcOps` implementation.
    unsafe fn is_unique(ref_counts: &RefCounts) -> bool;

    /// Determines how to make a mutable reference from a `RawRc`:
    ///
    /// - If both strong count and weak count are 1, returns `None`.
    /// - If strong count is 1 and weak count is greater than 1, returns
    ///   `Some(MakeMutStrategy::Move)`.
    /// - If strong count is greater than 1, returns `Some(MakeMutStrategy::Clone)`.
    ///
    /// # Safety
    ///
    /// - `ref_counts` should be provided by a `RawRc` object.
    /// - `ref_counts` should only be handled by the same `RcOps` implementation.
    #[cfg(not(no_global_oom_handling))]
    unsafe fn make_mut(ref_counts: &RefCounts) -> Option<MakeMutStrategy>;
}

/// Defines the `RefCounts` struct to store reference counts. The reference counters have enough
/// alignment to be operated atomically.
macro_rules! define_ref_counts {
    ($($target_pointer_width:literal => $align:literal,)*) => {
        $(
            /// Stores reference counts.
            #[cfg(target_pointer_width = $target_pointer_width)]
            #[repr(C, align($align))]
            pub(crate) struct RefCounts {
                /// Weak reference count (plus one if there are non-zero strong reference counts).
                pub(crate) weak: UnsafeCell<usize>,
                /// Strong reference count.
                pub(crate) strong: UnsafeCell<usize>,
            }
        )*
    };
}

// This ensures reference counters have correct alignment so that they can be treated as atomic
// reference counters for `Arc`.
define_ref_counts! {
    "16" => 2,
    "32" => 4,
    "64" => 8,
}

impl RefCounts {
    /// Creates a `RefCounts` with weak count of `1` and strong count of `strong_count`.
    pub(crate) const fn new(strong_count: usize) -> Self {
        Self { weak: UnsafeCell::new(1), strong: UnsafeCell::new(strong_count) }
    }
}

// Ensures the memory layout calculation works. (See module documentation.)
const _: () = assert!(size_of::<RefCounts>().is_power_of_two());

#[cfg(not(no_global_oom_handling))]
fn handle_layout_error<T>(result: Result<T, LayoutError>) -> T {
    result.unwrap()
}

/// A `Layout` that describes a reference-counted allocation.
#[derive(Clone, Copy)]
struct RcLayout(Layout);

impl RcLayout {
    /// Tries to create an `RcLayout` to store a value with layout `value_layout`. Returns `Err` if
    /// `value_layout` is too big to store in a reference-counted allocation.
    #[inline]
    const fn try_from_value_layout(value_layout: Layout) -> Result<Self, LayoutError> {
        match RefCounts::LAYOUT.extend(value_layout) {
            Ok((rc_layout, _)) => Ok(Self(rc_layout)),
            Err(error) => Err(error),
        }
    }

    /// Creates an `RcLayout` to store a value with layout `value_layout`. Panics if `value_layout`
    /// is too big to store in a reference-counted allocation.
    #[cfg(not(no_global_oom_handling))]
    #[inline]
    fn from_value_layout(value_layout: Layout) -> Self {
        handle_layout_error(Self::try_from_value_layout(value_layout))
    }

    /// Creates an `RcLayout` to store a value with layout `value_layout`.
    ///
    /// # Safety
    ///
    /// `RcLayout::try_from_value_layout(value_layout)` must return `Ok`.
    #[inline]
    unsafe fn from_value_layout_unchecked(value_layout: Layout) -> Self {
        let align = align_of::<RefCounts>().max(value_layout.align());
        let value_offset = size_of::<RefCounts>().max(value_layout.align());
        let size = unsafe { value_offset.unchecked_add(value_layout.size()) };

        Self(unsafe { Layout::from_size_align_unchecked(size, align) })
    }

    #[cfg(not(no_global_oom_handling))]
    fn try_new_array<T>(length: usize) -> Result<Self, LayoutError> {
        let value_layout = Layout::array::<T>(length)?;

        Self::try_from_value_layout(value_layout)
    }

    #[cfg(not(no_global_oom_handling))]
    fn new_array<T>(length: usize) -> Self {
        handle_layout_error(Self::try_new_array::<T>(length))
    }

    /// Returns the byte offset of the value stored in a reference-counted allocation that is
    /// described by `self`.
    #[inline]
    fn value_offset(&self) -> usize {
        size_of::<RefCounts>().max(self.align())
    }

    /// Returns the byte size of the value stored in a reference-counted allocation that is
    /// described by `self`.
    #[cfg(not(no_global_oom_handling))]
    #[inline]
    fn value_size(&self) -> usize {
        unsafe { self.size().unchecked_sub(self.value_offset()) }
    }

    /// Creates an `RcLayout` for storing a value that is pointed to by `value_ptr`.
    ///
    /// # Safety
    ///
    /// `value_ptr` has correct metadata of `T`.
    #[cfg(not(no_global_oom_handling))]
    unsafe fn from_value_ptr<T>(value_ptr: NonNull<T>) -> Self
    where
        T: ?Sized,
    {
        /// A helper trait for computing `RcLayout` to store a `Self` object. If `Self` is
        /// `Sized`, the `RcLayout` value is computed at compile time.
        trait SpecRcLayout {
            unsafe fn spec_rc_layout(value_ptr: NonNull<Self>) -> RcLayout;
        }

        impl<T> SpecRcLayout for T
        where
            T: ?Sized,
        {
            #[inline]
            default unsafe fn spec_rc_layout(value_ptr: NonNull<Self>) -> RcLayout {
                RcLayout::from_value_layout(unsafe { Layout::for_value_raw(value_ptr.as_ptr()) })
            }
        }

        impl<T> SpecRcLayout for T {
            #[inline]
            unsafe fn spec_rc_layout(_: NonNull<Self>) -> RcLayout {
                Self::RC_LAYOUT
            }
        }

        unsafe { T::spec_rc_layout(value_ptr) }
    }

    /// Creates an `RcLayout` for storing a value that is pointed to by `value_ptr`, assuming the
    /// value is small enough to fit inside a reference-counted allocation.
    ///
    /// # Safety
    ///
    /// - `value_ptr` has correct metadata for a `T` object.
    /// - It is known that the memory layout described by `value_ptr` can be used to create an
    ///   `RcLayout` successfully.
    unsafe fn from_value_ptr_unchecked<T>(value_ptr: NonNull<T>) -> Self
    where
        T: ?Sized,
    {
        /// A helper trait for computing `RcLayout` to store a `Self` object. If `Self` is
        /// `Sized`, the `RcLayout` value is computed at compile time.
        trait SpecRcLayoutUnchecked {
            unsafe fn spec_rc_layout_unchecked(value_ptr: NonNull<Self>) -> RcLayout;
        }

        impl<T> SpecRcLayoutUnchecked for T
        where
            T: ?Sized,
        {
            #[inline]
            default unsafe fn spec_rc_layout_unchecked(value_ptr: NonNull<Self>) -> RcLayout {
                unsafe {
                    RcLayout::from_value_layout_unchecked(Layout::for_value_raw(value_ptr.as_ptr()))
                }
            }
        }

        impl<T> SpecRcLayoutUnchecked for T {
            #[inline]
            unsafe fn spec_rc_layout_unchecked(_: NonNull<Self>) -> RcLayout {
                Self::RC_LAYOUT
            }
        }

        unsafe { T::spec_rc_layout_unchecked(value_ptr) }
    }
}

impl Deref for RcLayout {
    type Target = Layout;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

trait RcLayoutExt {
    /// Computes `RcLayout` at compile time if `Self` is `Sized`.
    const RC_LAYOUT: RcLayout;
}

impl<T> RcLayoutExt for T {
    const RC_LAYOUT: RcLayout = match RcLayout::try_from_value_layout(T::LAYOUT) {
        Ok(rc_layout) => rc_layout,
        Err(_) => panic!("value is too big to store in a reference-counted allocation"),
    };
}

/// Get a pointer to the `RefCounts` object in the same allocation with a value pointed to by
/// `value_ptr`.
///
/// # Safety
///
/// - `value_ptr` must point to a value object (can be uninitialized or dropped) that lives in a
///   reference-counted allocation.
unsafe fn ref_counts_ptr_from_value_ptr(value_ptr: NonNull<()>) -> NonNull<RefCounts> {
    const REF_COUNTS_OFFSET: usize = size_of::<RefCounts>();

    unsafe { value_ptr.byte_sub(REF_COUNTS_OFFSET) }.cast()
}

/// Get a pointer to the strong counter object in the same allocation with a value pointed to by
/// `value_ptr`.
///
/// # Safety
///
/// - `value_ptr` must point to a value object (can be uninitialized or dropped) that lives in a
///   reference-counted allocation.
unsafe fn strong_count_ptr_from_value_ptr(value_ptr: NonNull<()>) -> NonNull<UnsafeCell<usize>> {
    const STRONG_OFFSET: usize = size_of::<RefCounts>() - mem::offset_of!(RefCounts, strong);

    unsafe { value_ptr.byte_sub(STRONG_OFFSET) }.cast()
}

/// Get a pointer to the weak counter object in the same allocation with a value pointed to by
/// `value_ptr`.
///
/// # Safety
///
/// - `value_ptr` must point to a value object (can be uninitialized or dropped) that lives in a
///   reference-counted allocation.
unsafe fn weak_count_ptr_from_value_ptr(value_ptr: NonNull<()>) -> NonNull<UnsafeCell<usize>> {
    const WEAK_OFFSET: usize = size_of::<RefCounts>() - mem::offset_of!(RefCounts, weak);

    unsafe { value_ptr.byte_sub(WEAK_OFFSET) }.cast()
}

/// Initialize reference counters in a reference-counted allocation pointed to by `rc_ptr`
/// with a strong count of `STRONG_COUNT` and a weak count of 1.
///
/// # Safety
///
/// - `rc_ptr` points to a valid reference-counted allocation.
/// - `rc_layout` correctly describes the memory layout of the reference-counted allocation.
#[inline]
unsafe fn init_rc_allocation<const STRONG_COUNT: usize>(
    rc_ptr: NonNull<[u8]>,
    rc_layout: RcLayout,
) -> NonNull<()> {
    let ref_counts = const { RefCounts::new(STRONG_COUNT) };
    let rc_ptr = rc_ptr.cast::<()>();
    let value_offset = rc_layout.value_offset();
    let value_ptr = unsafe { rc_ptr.byte_add(value_offset) };

    unsafe { ref_counts_ptr_from_value_ptr(value_ptr).write(ref_counts) };

    value_ptr
}

/// If `allocation_result` is `Ok`, initialize the reference counts with strong count of
/// `STRONG_COUNT` and weak count of 1 and returns `Ok` with a pointer to the value object,
/// otherwise return the original error.
unsafe fn try_handle_rc_allocation_result<const STRONG_COUNT: usize>(
    allocation_result: Result<NonNull<[u8]>, AllocError>,
    rc_layout: RcLayout,
) -> Result<NonNull<()>, AllocError> {
    allocation_result.map(|rc_ptr| unsafe { init_rc_allocation::<STRONG_COUNT>(rc_ptr, rc_layout) })
}

/// Try to allocate a chunk of reference-counted memory that is described by `rc_layout` with
/// `alloc`. The allocated memory has strong count of `STRONG_COUNT` and weak count of 1.
fn try_allocate_uninit_in<A, const STRONG_COUNT: usize>(
    alloc: &A,
    rc_layout: RcLayout,
) -> Result<NonNull<()>, AllocError>
where
    A: Allocator,
{
    let allocation_result = alloc.allocate(*rc_layout);

    unsafe { try_handle_rc_allocation_result::<STRONG_COUNT>(allocation_result, rc_layout) }
}

/// Creates an allocator of type `A`, then tries to allocate a chunk of reference-counted memory
/// that is described by `rc_layout`.
fn try_allocate_uninit<A, const STRONG_COUNT: usize>(
    rc_layout: RcLayout,
) -> Result<(NonNull<()>, A), AllocError>
where
    A: Allocator + Default,
{
    let alloc = A::default();

    try_allocate_uninit_in::<A, STRONG_COUNT>(&alloc, rc_layout).map(|value_ptr| (value_ptr, alloc))
}

/// Tries to allocate a reference-counted memory that is described by `rc_layout` with `alloc`. The
/// allocated memory has strong count of `STRONG_COUNT` and weak count of 1, and the value memory
/// is all zero bytes.
fn try_allocate_zeroed_in<A, const STRONG_COUNT: usize>(
    alloc: &A,
    rc_layout: RcLayout,
) -> Result<NonNull<()>, AllocError>
where
    A: Allocator,
{
    let allocation_result = alloc.allocate_zeroed(*rc_layout);

    unsafe { try_handle_rc_allocation_result::<STRONG_COUNT>(allocation_result, rc_layout) }
}

/// Creates an allocator of type `A`, then tries to allocate a chunk of reference-counted memory with
/// all zero bytes memory that is described by `rc_layout`.
fn try_allocate_zeroed<A, const STRONG_COUNT: usize>(
    rc_layout: RcLayout,
) -> Result<(NonNull<()>, A), AllocError>
where
    A: Allocator + Default,
{
    let alloc = A::default();

    try_allocate_zeroed_in::<A, STRONG_COUNT>(&alloc, rc_layout).map(|value_ptr| (value_ptr, alloc))
}

/// If `allocation_result` is `Ok`, initialize the reference counts with strong count
/// `STRONG_COUNT` and weak count of 1 and returns a pointer to the value object, otherwise panic
/// will be triggered by calling `alloc::handle_alloc_error`.
#[cfg(not(no_global_oom_handling))]
#[inline]
unsafe fn handle_rc_allocation_result<const STRONG_COUNT: usize>(
    allocation_result: Result<NonNull<[u8]>, AllocError>,
    rc_layout: RcLayout,
) -> NonNull<()> {
    match allocation_result {
        Ok(rc_ptr) => unsafe { init_rc_allocation::<STRONG_COUNT>(rc_ptr, rc_layout) },
        Err(AllocError) => alloc::handle_alloc_error(*rc_layout),
    }
}

/// Allocates reference-counted memory that is described by `rc_layout` with `alloc`. The allocated
/// memory has strong count of `STRONG_COUNT` and weak count of 1. If the allocation fails, panic
/// will be triggered by calling `alloc::handle_alloc_error`.
#[cfg(not(no_global_oom_handling))]
#[inline]
fn allocate_uninit_in<A, const STRONG_COUNT: usize>(alloc: &A, rc_layout: RcLayout) -> NonNull<()>
where
    A: Allocator,
{
    let allocation_result = alloc.allocate(*rc_layout);

    unsafe { handle_rc_allocation_result::<STRONG_COUNT>(allocation_result, rc_layout) }
}

/// Creates an allocator of type `A`, then allocate a chunk of reference-counted memory that is
/// described by `rc_layout`.
#[cfg(not(no_global_oom_handling))]
#[inline]
fn allocate_uninit<A, const STRONG_COUNT: usize>(rc_layout: RcLayout) -> (NonNull<()>, A)
where
    A: Allocator + Default,
{
    let alloc = A::default();
    let value_ptr = allocate_uninit_in::<A, STRONG_COUNT>(&alloc, rc_layout);

    (value_ptr, alloc)
}

/// Allocates reference-counted memory that is described by `rc_layout` with `alloc`. The allocated
/// memory has strong count of `STRONG_COUNT` and weak count of 1, and the value memory is all zero
/// bytes. If the allocation fails, panic will be triggered by calling `alloc::handle_alloc_error`.
#[cfg(not(no_global_oom_handling))]
fn allocate_zeroed_in<A, const STRONG_COUNT: usize>(alloc: &A, rc_layout: RcLayout) -> NonNull<()>
where
    A: Allocator,
{
    let allocation_result = alloc.allocate_zeroed(*rc_layout);

    unsafe { handle_rc_allocation_result::<STRONG_COUNT>(allocation_result, rc_layout) }
}

/// Creates an allocator of type `A`, then allocate a chunk of reference-counted memory with all
/// zero bytes that is described by `rc_layout`.
#[cfg(not(no_global_oom_handling))]
fn allocate_zeroed<A, const STRONG_COUNT: usize>(rc_layout: RcLayout) -> (NonNull<()>, A)
where
    A: Allocator + Default,
{
    let alloc = A::default();
    let value_ptr = allocate_zeroed_in::<A, STRONG_COUNT>(&alloc, rc_layout);

    (value_ptr, alloc)
}

/// Allocate a reference-counted memory chunk for storing a value according to `rc_layout`, then
/// initialize the value with `f`. If `f` panics, the memory will be freed.
#[cfg(not(no_global_oom_handling))]
fn allocate_with_in<A, F, const STRONG_COUNT: usize>(
    alloc: &A,
    rc_layout: RcLayout,
    f: F,
) -> NonNull<()>
where
    A: Allocator,
    F: FnOnce(NonNull<()>),
{
    struct Guard<'a, A>
    where
        A: Allocator,
    {
        value_ptr: NonNull<()>,
        alloc: &'a A,
        rc_layout: RcLayout,
    }

    impl<'a, A> Drop for Guard<'a, A>
    where
        A: Allocator,
    {
        fn drop(&mut self) {
            unsafe { deallocate_value_ptr::<A>(self.value_ptr, self.alloc, self.rc_layout) };
        }
    }

    let value_ptr = allocate_uninit_in::<A, STRONG_COUNT>(alloc, rc_layout);
    let guard = Guard { value_ptr, alloc, rc_layout };

    f(value_ptr);

    mem::forget(guard);

    value_ptr
}

/// Creates an allocator of type `A`, then allocate a chunk of reference-counted memory that is
/// described by `rc_layout`. `f` will be called with a pointer that points the value storage to
/// initialize the allocated memory. If `f` panics, the allocated memory will be deallocated.
#[cfg(not(no_global_oom_handling))]
fn allocate_with<A, F, const STRONG_COUNT: usize>(rc_layout: RcLayout, f: F) -> (NonNull<()>, A)
where
    A: Allocator + Default,
    F: FnOnce(NonNull<()>),
{
    let alloc = A::default();
    let value_ptr = allocate_with_in::<A, F, STRONG_COUNT>(&alloc, rc_layout, f);

    (value_ptr, alloc)
}

/// Allocate reference-counted memory that has strong count of `STRONG_COUNT` and weak count of 1.
/// The value will be initialized with data pointed to by `src_ptr`.
///
/// # Safety
///
/// - Memory pointed to by `src_ptr` has enough data to read for filling the value in an allocation
///   that is described by `rc_layout`.
#[cfg(not(no_global_oom_handling))]
unsafe fn allocate_with_bytes_in<A, const STRONG_COUNT: usize>(
    src_ptr: NonNull<()>,
    alloc: &A,
    rc_layout: RcLayout,
) -> NonNull<()>
where
    A: Allocator,
{
    let value_ptr = allocate_uninit_in::<A, STRONG_COUNT>(alloc, rc_layout);
    let value_size = rc_layout.value_size();

    unsafe {
        ptr::copy_nonoverlapping::<u8>(
            src_ptr.as_ptr().cast(),
            value_ptr.as_ptr().cast(),
            value_size,
        )
    };

    value_ptr
}

/// Like `<*mut T>::with_metadata_of::<U>`, but for `NonNull<T>`.
unsafe fn non_null_with_metadata_of<T, U>(ptr: NonNull<T>, meta: NonNull<U>) -> NonNull<U>
where
    T: ?Sized,
    U: ?Sized,
{
    unsafe { NonNull::new_unchecked(ptr.as_ptr().with_metadata_of(meta.as_ptr())) }
}

/// Allocate a chunk of reference-counted memory with a value that is copied from `value`.
#[cfg(not(no_global_oom_handling))]
fn allocate_with_value_in<T, A, const STRONG_COUNT: usize>(src: &T, alloc: &A) -> NonNull<T>
where
    T: ?Sized,
    A: Allocator,
{
    let src_ptr = NonNull::from(src);
    let rc_layout = unsafe { RcLayout::from_value_ptr(src_ptr) };

    let value_ptr =
        unsafe { allocate_with_bytes_in::<A, STRONG_COUNT>(src_ptr.cast(), alloc, rc_layout) };

    unsafe { non_null_with_metadata_of(value_ptr, src_ptr) }
}

/// Creates an allocator of type `A`, then allocates a chunk of reference-counted memory with value
/// copied from `value`.
#[cfg(not(no_global_oom_handling))]
#[inline]
fn allocate_with_value<T, A, const STRONG_COUNT: usize>(value: &T) -> (NonNull<T>, A)
where
    T: ?Sized,
    A: Allocator + Default,
{
    let alloc = A::default();
    let value_ptr = allocate_with_value_in::<T, A, STRONG_COUNT>(value, &alloc);

    (value_ptr, alloc)
}

/// Deallocate a reference-counted allocation with a value object pointed to by `value_ptr`.
#[inline]
unsafe fn deallocate_value_ptr<A>(value_ptr: NonNull<()>, alloc: &A, rc_layout: RcLayout)
where
    A: Allocator,
{
    let value_offset = rc_layout.value_offset();
    let rc_ptr = unsafe { value_ptr.byte_sub(value_offset) };

    unsafe { alloc.deallocate(rc_ptr.cast(), *rc_layout) }
}

#[inline]
fn is_dangling(value_ptr: NonNull<()>) -> bool {
    value_ptr.addr() == NonZeroUsize::MAX
}

/// Decrement strong reference count in a reference-counted allocation with a value object that is
/// pointed to by `value_ptr`.
unsafe fn decrement_strong_ref_count<R>(value_ptr: NonNull<()>) -> bool
where
    R: RcOps,
{
    unsafe { R::decrement_ref_count(strong_count_ptr_from_value_ptr(value_ptr).as_ref()) }
}

/// Decrement weak reference count in a reference-counted allocation with a value object that is
/// pointed to by `value_ptr`.
unsafe fn decrement_weak_ref_count<R>(value_ptr: NonNull<()>) -> bool
where
    R: RcOps,
{
    unsafe { R::decrement_ref_count(weak_count_ptr_from_value_ptr(value_ptr).as_ref()) }
}

/// Increment strong reference count in a reference-counted allocation with a value object that is
/// pointed to by `value_ptr`.
unsafe fn increment_strong_ref_count<R>(value_ptr: NonNull<()>)
where
    R: RcOps,
{
    unsafe { R::increment_ref_count(strong_count_ptr_from_value_ptr(value_ptr).as_ref()) }
}

/// Increment weak reference count in a reference-counted allocation with a value object that is
/// pointed to by `value_ptr`.
unsafe fn increment_weak_ref_count<R>(value_ptr: NonNull<()>)
where
    R: RcOps,
{
    unsafe { R::increment_ref_count(weak_count_ptr_from_value_ptr(value_ptr).as_ref()) }
}

/// Calls `RawWeak::drop_unchecked` on drop.
struct WeakGuard<'a, T, A, R>
where
    T: ?Sized,
    A: Allocator,
    R: RcOps,
{
    weak: &'a mut RawWeak<T, A>,
    _phantom_data: PhantomData<R>,
}

impl<'a, T, A, R> WeakGuard<'a, T, A, R>
where
    T: ?Sized,
    A: Allocator,
    R: RcOps,
{
    /// # Safety
    ///
    /// - `weak` is non-dangling.
    /// - After `WeakGuard` being dropped, the content pointed to by `weak.ptr` should not be accessed
    ///   anymore.
    unsafe fn new(weak: &'a mut RawWeak<T, A>) -> Self {
        Self { weak, _phantom_data: PhantomData }
    }
}

impl<T, A, R> Drop for WeakGuard<'_, T, A, R>
where
    T: ?Sized,
    A: Allocator,
    R: RcOps,
{
    fn drop(&mut self) {
        unsafe { self.weak.drop_unchecked::<R>() };
    }
}

/// Calls `RawRc::drop` on drop.
struct RcGuard<'a, T, A, R>
where
    T: ?Sized,
    A: Allocator,
    R: RcOps,
{
    rc: &'a mut RawRc<T, A>,
    _phantom_data: PhantomData<R>,
}

impl<'a, T, A, R> RcGuard<'a, T, A, R>
where
    T: ?Sized,
    A: Allocator,
    R: RcOps,
{
    /// # Safety
    ///
    /// - After `WeakGuard` being dropped, the content pointed to by `rc.weak.ptr` should not be
    ///   accessed anymore.
    unsafe fn new(rc: &'a mut RawRc<T, A>) -> Self {
        Self { rc, _phantom_data: PhantomData }
    }
}

impl<T, A, R> Drop for RcGuard<'_, T, A, R>
where
    T: ?Sized,
    A: Allocator,
    R: RcOps,
{
    fn drop(&mut self) {
        unsafe { self.rc.drop::<R>() };
    }
}

/// Base implementation of a weak pointer. `RawWeak` does not implement `Drop`, user should call
/// `RawWeak::drop` or `RawWeak::drop_unchecked` manually to drop this object.
///
/// A `RawWeak` can be either dangling or non-dangling. A dangling `RawWeak` does not point to a
/// valid value. A non-dangling `RawWeak` points to a valid reference-counted allocation. The value
/// pointed to by a `RawWeak` may be uninitialized.
pub(crate) struct RawWeak<T, A>
where
    T: ?Sized,
{
    /// Points to a (possibly uninitialized or dropped) `T` value inside of a reference-counted
    /// allocation.
    ptr: NonNull<T>,

    /// The allocator for `ptr`.
    alloc: A,
}

impl<T, A> RawWeak<T, A>
where
    T: ?Sized,
{
    pub(crate) const unsafe fn from_raw_parts(ptr: NonNull<T>, alloc: A) -> Self {
        Self { ptr, alloc }
    }

    pub(crate) unsafe fn from_raw(ptr: NonNull<T>) -> Self
    where
        A: Default,
    {
        unsafe { Self::from_raw_parts(ptr, A::default()) }
    }

    pub(crate) fn allocator(&self) -> &A {
        &self.alloc
    }

    pub(crate) fn as_ptr(&self) -> NonNull<T> {
        self.ptr
    }

    #[inline(never)]
    unsafe fn assume_init_drop_slow<R>(&mut self)
    where
        A: Allocator,
        R: RcOps,
    {
        let guard = unsafe { WeakGuard::<T, A, R>::new(self) };

        unsafe { guard.weak.ptr.drop_in_place() };
    }

    /// Assume the value pointed to by `ptr` is initialized, drop the value along the `RawWeak` object.
    #[inline]
    unsafe fn assume_init_drop<R>(&mut self)
    where
        A: Allocator,
        R: RcOps,
    {
        if const { mem::needs_drop::<T>() } {
            unsafe { self.assume_init_drop_slow::<R>() };
        } else {
            unsafe { self.drop_unchecked::<R>() };
        }
    }

    pub(crate) unsafe fn cast<U>(self) -> RawWeak<U, A> {
        unsafe { self.cast_with(NonNull::cast) }
    }

    #[inline]
    pub(crate) unsafe fn cast_with<U, F>(self, f: F) -> RawWeak<U, A>
    where
        U: ?Sized,
        F: FnOnce(NonNull<T>) -> NonNull<U>,
    {
        unsafe { RawWeak::from_raw_parts(f(self.ptr), self.alloc) }
    }

    #[inline]
    pub(crate) unsafe fn clone<R>(&self) -> Self
    where
        A: Clone,
        R: RcOps,
    {
        unsafe fn inner<R>(ptr: NonNull<()>)
        where
            R: RcOps,
        {
            if !is_dangling(ptr) {
                unsafe { increment_weak_ref_count::<R>(ptr) };
            }
        }

        unsafe {
            inner::<R>(self.ptr.cast());

            Self::from_raw_parts(self.ptr, self.alloc.clone())
        }
    }

    /// Drop this weak pointer.
    #[inline]
    pub(crate) unsafe fn drop<R>(&mut self)
    where
        A: Allocator,
        R: RcOps,
    {
        if !is_dangling(self.ptr.cast()) {
            unsafe { self.drop_unchecked::<R>() };
        }
    }

    /// Drop this weak pointer, assume `self` is non-dangling.
    #[inline]
    unsafe fn drop_unchecked<R>(&mut self)
    where
        A: Allocator,
        R: RcOps,
    {
        let is_last_weak_ref = unsafe { decrement_weak_ref_count::<R>(self.ptr.cast()) };

        if is_last_weak_ref {
            let rc_layout = unsafe { RcLayout::from_value_ptr_unchecked(self.ptr) };

            unsafe { deallocate_value_ptr::<A>(self.ptr.cast(), &self.alloc, rc_layout) }
        }
    }

    pub(crate) fn into_raw(self) -> NonNull<T> {
        self.ptr
    }

    pub(crate) fn into_raw_parts(self) -> (NonNull<T>, A) {
        (self.ptr, self.alloc)
    }

    pub(crate) fn ptr_eq(&self, other: &Self) -> bool {
        ptr::addr_eq(self.ptr.as_ptr(), other.ptr.as_ptr())
    }

    pub(crate) fn ptr_ne(&self, other: &Self) -> bool {
        !ptr::addr_eq(self.ptr.as_ptr(), other.ptr.as_ptr())
    }

    /// Returns the `RefCounts` object inside the reference-counted allocation if `self` is
    /// non-dangling.
    #[cfg(not(no_sync))]
    pub(crate) fn ref_counts(&self) -> Option<&RefCounts> {
        (!is_dangling(self.ptr.cast())).then(|| unsafe { self.ref_counts_unchecked() })
    }

    /// Returns the `RefCounts` object inside the reference-counted allocation, assume `self` is
    /// non-dangling.
    ///
    /// # Safety
    ///
    /// `self` is non-dangling.
    unsafe fn ref_counts_unchecked(&self) -> &RefCounts {
        unsafe { ref_counts_ptr_from_value_ptr(self.ptr.cast()).as_ref() }
    }

    /// Returns the strong reference count object inside the reference-counted allocation if `self`
    /// is non-dangling.
    pub(crate) fn strong_count(&self) -> Option<&UnsafeCell<usize>> {
        (!is_dangling(self.ptr.cast())).then(|| unsafe { self.strong_count_unchecked() })
    }

    /// Returns the strong reference count object inside the reference-counted allocation, assume
    /// `self` is non-dangling.
    ///
    /// # Safety
    ///
    /// `self` is non-dangling.
    unsafe fn strong_count_unchecked(&self) -> &UnsafeCell<usize> {
        unsafe { strong_count_ptr_from_value_ptr(self.ptr.cast()).as_ref() }
    }

    /// Returns the weak reference count object inside the reference-counted allocation if `self`
    /// is non-dangling.
    pub(crate) fn weak_count(&self) -> Option<&UnsafeCell<usize>> {
        (!is_dangling(self.ptr.cast())).then(|| unsafe { self.weak_count_unchecked() })
    }

    /// Returns the weak reference count object inside the reference-counted allocation, assume
    /// `self` is non-dangling.
    ///
    /// # Safety
    ///
    /// `self` is non-dangling.
    unsafe fn weak_count_unchecked(&self) -> &UnsafeCell<usize> {
        unsafe { weak_count_ptr_from_value_ptr(self.ptr.cast()).as_ref() }
    }

    /// Creates a `RawRc` object if there are non-zero strong reference counts.
    ///
    /// # Safety
    ///
    /// `self` should only be handled by the same `RcOps` implementation.
    pub(crate) unsafe fn upgrade<R>(&self) -> Option<RawRc<T, A>>
    where
        A: Clone,
        R: RcOps,
    {
        unsafe fn inner<R>(value_ptr: NonNull<()>) -> bool
        where
            R: RcOps,
        {
            (!is_dangling(value_ptr))
                && unsafe { R::upgrade(strong_count_ptr_from_value_ptr(value_ptr).as_ref()) }
        }

        unsafe {
            inner::<R>(self.ptr.cast()).then(|| RawRc::from_raw_parts(self.ptr, self.alloc.clone()))
        }
    }
}

impl<T, A> RawWeak<T, A> {
    pub(crate) fn new_dangling() -> Self
    where
        A: Default,
    {
        Self::new_dangling_in(A::default())
    }

    pub(crate) const fn new_dangling_in(alloc: A) -> Self {
        unsafe { Self::from_raw_parts(NonNull::without_provenance(NonZeroUsize::MAX), alloc) }
    }

    pub(crate) fn try_new_uninit<const STRONG_COUNT: usize>() -> Result<Self, AllocError>
    where
        A: Allocator + Default,
    {
        try_allocate_uninit::<A, STRONG_COUNT>(T::RC_LAYOUT)
            .map(|(ptr, alloc)| unsafe { Self::from_raw_parts(ptr.cast(), alloc) })
    }

    pub(crate) fn try_new_uninit_in<const STRONG_COUNT: usize>(alloc: A) -> Result<Self, AllocError>
    where
        A: Allocator,
    {
        try_allocate_uninit_in::<A, STRONG_COUNT>(&alloc, T::RC_LAYOUT)
            .map(|ptr| unsafe { Self::from_raw_parts(ptr.cast(), alloc) })
    }

    pub(crate) fn try_new_zeroed<const STRONG_COUNT: usize>() -> Result<Self, AllocError>
    where
        A: Allocator + Default,
    {
        try_allocate_zeroed::<A, STRONG_COUNT>(T::RC_LAYOUT)
            .map(|(ptr, alloc)| unsafe { Self::from_raw_parts(ptr.cast(), alloc) })
    }

    pub(crate) fn try_new_zeroed_in<const STRONG_COUNT: usize>(alloc: A) -> Result<Self, AllocError>
    where
        A: Allocator,
    {
        try_allocate_zeroed_in::<A, STRONG_COUNT>(&alloc, T::RC_LAYOUT)
            .map(|ptr| unsafe { Self::from_raw_parts(ptr.cast(), alloc) })
    }

    #[cfg(not(no_global_oom_handling))]
    pub(crate) fn new_uninit<const STRONG_COUNT: usize>() -> Self
    where
        A: Allocator + Default,
    {
        let (ptr, alloc) = allocate_uninit::<A, STRONG_COUNT>(T::RC_LAYOUT);

        unsafe { Self::from_raw_parts(ptr.cast(), alloc) }
    }

    #[cfg(not(no_global_oom_handling))]
    pub(crate) fn new_uninit_in<const STRONG_COUNT: usize>(alloc: A) -> Self
    where
        A: Allocator,
    {
        unsafe {
            Self::from_raw_parts(
                allocate_uninit_in::<A, STRONG_COUNT>(&alloc, T::RC_LAYOUT).cast(),
                alloc,
            )
        }
    }

    #[cfg(not(no_global_oom_handling))]
    pub(crate) fn new_zeroed<const STRONG_COUNT: usize>() -> Self
    where
        A: Allocator + Default,
    {
        let (ptr, alloc) = allocate_zeroed::<A, STRONG_COUNT>(T::RC_LAYOUT);

        unsafe { Self::from_raw_parts(ptr.cast(), alloc) }
    }

    #[cfg(not(no_global_oom_handling))]
    pub(crate) fn new_zeroed_in<const STRONG_COUNT: usize>(alloc: A) -> Self
    where
        A: Allocator,
    {
        unsafe {
            Self::from_raw_parts(
                allocate_zeroed_in::<A, STRONG_COUNT>(&alloc, T::RC_LAYOUT).cast(),
                alloc,
            )
        }
    }

    /// Take the value pointed to by `self` and drop the `RawWeak` object.
    ///
    /// # Safety
    ///
    /// - `self` is non-dangling.
    /// - The value pointed to by `self` is initialized.
    /// - The strong reference count is zero.
    unsafe fn assume_init_into_inner<R>(mut self) -> T
    where
        A: Allocator,
        R: RcOps,
    {
        unsafe {
            let result = self.ptr.read();

            self.drop_unchecked::<R>();

            result
        }
    }
}

impl<T, A> RawWeak<[T], A> {
    #[cfg(not(no_global_oom_handling))]
    fn allocate<F>(length: usize, allocate_fn: F) -> Self
    where
        A: Allocator,
        F: FnOnce(RcLayout) -> (NonNull<()>, A),
    {
        let rc_layout = RcLayout::new_array::<T>(length);
        let (ptr, alloc) = allocate_fn(rc_layout);

        unsafe { Self::from_raw_parts(NonNull::slice_from_raw_parts(ptr.cast(), length), alloc) }
    }

    #[cfg(not(no_global_oom_handling))]
    fn allocate_in<F>(length: usize, alloc: A, allocate_fn: F) -> Self
    where
        A: Allocator,
        F: FnOnce(&A, RcLayout) -> NonNull<()>,
    {
        let rc_layout = RcLayout::new_array::<T>(length);
        let ptr = allocate_fn(&alloc, rc_layout);

        unsafe { Self::from_raw_parts(NonNull::slice_from_raw_parts(ptr.cast(), length), alloc) }
    }

    #[cfg(not(no_global_oom_handling))]
    pub(crate) fn new_uninit_slice<const STRONG_COUNT: usize>(length: usize) -> Self
    where
        A: Allocator + Default,
    {
        Self::allocate(length, allocate_uninit::<A, STRONG_COUNT>)
    }

    #[cfg(not(no_global_oom_handling))]
    pub(crate) fn new_uninit_slice_in<const STRONG_COUNT: usize>(length: usize, alloc: A) -> Self
    where
        A: Allocator,
    {
        Self::allocate_in(length, alloc, allocate_uninit_in::<A, STRONG_COUNT>)
    }

    #[cfg(not(no_global_oom_handling))]
    pub(crate) fn new_zeroed_slice<const STRONG_COUNT: usize>(length: usize) -> Self
    where
        A: Allocator + Default,
    {
        Self::allocate(length, allocate_zeroed::<A, STRONG_COUNT>)
    }

    #[cfg(not(no_global_oom_handling))]
    pub(crate) fn new_zeroed_slice_in<const STRONG_COUNT: usize>(length: usize, alloc: A) -> Self
    where
        A: Allocator,
    {
        Self::allocate_in(length, alloc, allocate_zeroed_in::<A, STRONG_COUNT>)
    }
}

impl<T, U, A> CoerceUnsized<RawWeak<U, A>> for RawWeak<T, A>
where
    T: Unsize<U> + ?Sized,
    U: ?Sized,
{
}

impl<T, A> Debug for RawWeak<T, A>
where
    T: ?Sized,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str("(Weak)")
    }
}

impl<T, A> Default for RawWeak<T, A>
where
    A: Default,
{
    fn default() -> Self {
        Self::new_dangling()
    }
}

impl<T, U> DispatchFromDyn<RawWeak<U, Global>> for RawWeak<T, Global>
where
    T: Unsize<U> + ?Sized,
    U: ?Sized,
{
}

/// Base implementation of a strong pointer. `RawRc` does not implement `Drop`, user should call
/// `RawRc::drop` manually to drop this object.
#[repr(transparent)]
pub(crate) struct RawRc<T, A>
where
    T: ?Sized,
{
    /// A `RawRc` is just a `RawWeak` that has a strong reference count that is owned by the `RawRc`
    /// object. The weak pointer is always non-dangling.
    weak: RawWeak<T, A>,

    // Defines the ownership of `T` for drop-check.
    _phantom_data: PhantomData<T>,
}

impl<T, A> RawRc<T, A>
where
    T: ?Sized,
{
    /// # Safety
    ///
    /// - `ptr` points to a value inside a reference-counted allocation.
    /// - The allocation can be freed by `A::default()`.
    pub(crate) unsafe fn from_raw(ptr: NonNull<T>) -> Self
    where
        A: Default,
    {
        unsafe { Self::from_raw_parts(ptr, A::default()) }
    }

    /// # Safety
    ///
    /// - `ptr` points to a value inside a reference-counted allocation.
    /// - The allocation can be freed by `alloc`.
    pub(crate) unsafe fn from_raw_parts(ptr: NonNull<T>, alloc: A) -> Self {
        unsafe { Self::from_weak(RawWeak::from_raw_parts(ptr, alloc)) }
    }

    /// # Safety
    ///
    /// `weak` must have at least one unowned strong reference count. The newly created `RawRc` will
    /// take the ownership of exactly one strong reference count.
    unsafe fn from_weak(weak: RawWeak<T, A>) -> Self {
        Self { weak, _phantom_data: PhantomData }
    }

    pub(crate) fn allocator(&self) -> &A {
        &self.weak.alloc
    }

    pub(crate) fn as_ptr(&self) -> NonNull<T> {
        self.weak.ptr
    }

    pub(crate) unsafe fn cast<U>(self) -> RawRc<U, A> {
        unsafe { RawRc::from_weak(self.weak.cast()) }
    }

    #[inline]
    pub(crate) unsafe fn cast_with<U, F>(self, f: F) -> RawRc<U, A>
    where
        U: ?Sized,
        F: FnOnce(NonNull<T>) -> NonNull<U>,
    {
        unsafe { RawRc::from_weak(self.weak.cast_with(f)) }
    }

    #[inline]
    pub(crate) unsafe fn clone<R>(&self) -> Self
    where
        A: Clone,
        R: RcOps,
    {
        unsafe {
            increment_strong_ref_count::<R>(self.weak.ptr.cast());

            Self::from_raw_parts(self.weak.ptr, self.weak.alloc.clone())
        }
    }

    pub(crate) unsafe fn decrement_strong_count<R: RcOps>(ptr: NonNull<T>)
    where
        A: Allocator + Default,
    {
        unsafe { Self::decrement_strong_count_in::<R>(ptr, A::default()) };
    }

    pub(crate) unsafe fn decrement_strong_count_in<R: RcOps>(ptr: NonNull<T>, alloc: A)
    where
        A: Allocator,
    {
        unsafe { RawRc::from_raw_parts(ptr, alloc).drop::<R>() };
    }

    pub(crate) unsafe fn increment_strong_count<R: RcOps>(ptr: NonNull<T>) {
        unsafe { increment_strong_ref_count::<R>(ptr.cast()) };
    }

    pub(crate) unsafe fn downgrade<R>(&self) -> RawWeak<T, A>
    where
        A: Clone,
        R: RcOps,
    {
        unsafe fn inner<R>(value_ptr: NonNull<()>)
        where
            R: RcOps,
        {
            unsafe { R::downgrade(weak_count_ptr_from_value_ptr(value_ptr).as_ref()) };
        }

        unsafe {
            inner::<R>(self.weak.ptr.cast());

            RawWeak::from_raw_parts(self.weak.ptr, self.weak.alloc.clone())
        }
    }

    #[inline]
    pub(crate) unsafe fn drop<R>(&mut self)
    where
        A: Allocator,
        R: RcOps,
    {
        let is_last_strong_ref = unsafe { decrement_strong_ref_count::<R>(self.weak.ptr.cast()) };

        if is_last_strong_ref {
            unsafe { self.weak.assume_init_drop::<R>() }
        }
    }

    pub(crate) unsafe fn get_mut<R>(&mut self) -> Option<&mut T>
    where
        R: RcOps,
    {
        unsafe fn inner<R>(value_ptr: NonNull<()>) -> Option<NonNull<()>>
        where
            R: RcOps,
        {
            unsafe { R::is_unique(ref_counts_ptr_from_value_ptr(value_ptr).as_ref()) }
                .then_some(value_ptr)
        }

        unsafe { inner::<R>(self.weak.ptr.cast()) }
            .map(|ptr| unsafe { non_null_with_metadata_of(ptr, self.weak.ptr).as_mut() })
    }

    pub(crate) unsafe fn get_mut_unchecked(&mut self) -> &mut T {
        unsafe { self.weak.ptr.as_mut() }
    }

    pub(crate) fn into_raw(self) -> NonNull<T> {
        self.weak.into_raw()
    }

    pub(crate) fn into_raw_parts(self) -> (NonNull<T>, A) {
        self.weak.into_raw_parts()
    }

    #[cfg(not(no_global_oom_handling))]
    pub(crate) unsafe fn make_mut<R>(&mut self) -> &mut T
    where
        T: CloneToUninit,
        A: Allocator + Clone,
        R: RcOps,
    {
        struct SetRcPtrOnDrop<'a, T, A>
        where
            T: ?Sized,
        {
            rc: &'a mut RawRc<T, A>,
            new_ptr: NonNull<T>,
        }

        impl<T, A> Drop for SetRcPtrOnDrop<'_, T, A>
        where
            T: ?Sized,
        {
            fn drop(&mut self) {
                self.rc.weak.ptr = self.new_ptr;
            }
        }

        unsafe {
            if let Some(strategy) = R::make_mut(self.ref_counts()) {
                let rc_layout = RcLayout::from_value_ptr_unchecked(self.weak.ptr);

                match strategy {
                    MakeMutStrategy::Move => {
                        // `R::make_mut` has made strong reference count to zero, so the `RawRc`
                        // object is essentially a `RawWeak` object but has its value initialized.
                        // This means we are the only owner of the value and we can safely move the
                        // value into a new allocation.

                        // This guarantees to drop old `RawRc` object even if the allocation
                        // panics.
                        let guard = WeakGuard::<T, A, R>::new(&mut self.weak);

                        let new_ptr = allocate_with_bytes_in::<A, 1>(
                            guard.weak.ptr.cast(),
                            &guard.weak.alloc,
                            rc_layout,
                        );

                        // No panic happens, defuse the guard.

                        mem::forget(guard);

                        let new_ptr = non_null_with_metadata_of(new_ptr, self.weak.ptr);

                        // Ensure the value pointer in `self` is updated to `new_ptr`.
                        let update_ptr_on_drop = SetRcPtrOnDrop { rc: self, new_ptr };

                        // The strong count .
                        update_ptr_on_drop.rc.weak.drop_unchecked::<R>();
                    }
                    MakeMutStrategy::Clone => {
                        // There are multiple owners of the value, we need to clone the value into a
                        // new allocation.

                        let new_ptr =
                            allocate_with_in::<A, _, 1>(&self.weak.alloc, rc_layout, |dst_ptr| {
                                T::clone_to_uninit(self.as_ref(), dst_ptr.as_ptr().cast())
                            });

                        let new_ptr = non_null_with_metadata_of(new_ptr, self.weak.ptr);

                        // Ensure the value pointer in `self` is updated to `new_ptr`.
                        let update_ptr_on_drop = SetRcPtrOnDrop { rc: self, new_ptr };

                        // Manually drop old `RawRc`.
                        update_ptr_on_drop.rc.drop::<R>();
                    }
                }
            }

            self.get_mut_unchecked()
        }
    }

    pub(crate) fn ptr_eq(&self, other: &Self) -> bool {
        RawWeak::ptr_eq(&self.weak, &other.weak)
    }

    pub(crate) fn ptr_ne(&self, other: &Self) -> bool {
        RawWeak::ptr_ne(&self.weak, &other.weak)
    }

    #[cfg(not(no_global_oom_handling))]
    pub(crate) fn ref_counts(&self) -> &RefCounts {
        unsafe { self.weak.ref_counts_unchecked() }
    }

    pub(crate) fn strong_count(&self) -> &UnsafeCell<usize> {
        unsafe { self.weak.strong_count_unchecked() }
    }

    pub(crate) fn weak_count(&self) -> &UnsafeCell<usize> {
        unsafe { self.weak.weak_count_unchecked() }
    }
}

impl<T, A> RawRc<T, A> {
    unsafe fn from_weak_with_value(weak: RawWeak<T, A>, value: T) -> Self {
        unsafe {
            weak.ptr.write(value);

            Self::from_weak(weak)
        }
    }

    #[inline]
    pub(crate) fn try_new(value: T) -> Result<Self, AllocError>
    where
        A: Allocator + Default,
    {
        RawWeak::try_new_uninit::<1>()
            .map(|weak| unsafe { Self::from_weak_with_value(weak, value) })
    }

    #[inline]
    pub(crate) fn try_new_in(value: T, alloc: A) -> Result<Self, AllocError>
    where
        A: Allocator,
    {
        RawWeak::try_new_uninit_in::<1>(alloc)
            .map(|weak| unsafe { Self::from_weak_with_value(weak, value) })
    }

    #[cfg(not(no_global_oom_handling))]
    #[inline]
    pub(crate) fn new(value: T) -> Self
    where
        A: Allocator + Default,
    {
        unsafe { Self::from_weak_with_value(RawWeak::new_uninit::<1>(), value) }
    }

    #[cfg(not(no_global_oom_handling))]
    #[inline]
    pub(crate) fn new_in(value: T, alloc: A) -> Self
    where
        A: Allocator,
    {
        unsafe { Self::from_weak_with_value(RawWeak::new_uninit_in::<1>(alloc), value) }
    }

    #[cfg(not(no_global_oom_handling))]
    fn new_with<F>(f: F) -> Self
    where
        A: Allocator + Default,
        F: FnOnce() -> T,
    {
        let (ptr, alloc) =
            allocate_with::<A, _, 1>(T::RC_LAYOUT, |ptr| unsafe { ptr.cast().write(f()) });

        unsafe { Self::from_raw_parts(ptr.cast(), alloc) }
    }

    #[cfg(not(no_global_oom_handling))]
    unsafe fn new_cyclic_impl<F, R>(mut weak: RawWeak<T, A>, data_fn: F) -> Self
    where
        A: Allocator,
        F: FnOnce(&RawWeak<T, A>) -> T,
        R: RcOps,
    {
        let guard = unsafe { WeakGuard::<T, A, R>::new(&mut weak) };
        let data = data_fn(guard.weak);

        mem::forget(guard);

        unsafe { RawUniqueRc::from_weak_with_value(weak, data).into_rc::<R>() }
    }

    #[cfg(not(no_global_oom_handling))]
    pub(crate) unsafe fn new_cyclic<F, R>(data_fn: F) -> Self
    where
        A: Allocator + Default,
        F: FnOnce(&RawWeak<T, A>) -> T,
        R: RcOps,
    {
        let weak = RawWeak::new_uninit::<0>();

        unsafe { Self::new_cyclic_impl::<F, R>(weak, data_fn) }
    }

    #[cfg(not(no_global_oom_handling))]
    pub(crate) unsafe fn new_cyclic_in<F, R>(data_fn: F, alloc: A) -> Self
    where
        A: Allocator,
        F: FnOnce(&RawWeak<T, A>) -> T,
        R: RcOps,
    {
        let weak = RawWeak::new_uninit_in::<0>(alloc);

        unsafe { Self::new_cyclic_impl::<F, R>(weak, data_fn) }
    }

    pub(crate) unsafe fn into_inner<R>(self) -> Option<T>
    where
        A: Allocator,
        R: RcOps,
    {
        let is_last_strong_ref = unsafe { decrement_strong_ref_count::<R>(self.weak.ptr.cast()) };

        is_last_strong_ref.then(|| unsafe { self.weak.assume_init_into_inner::<R>() })
    }

    pub(crate) unsafe fn try_unwrap<R>(self) -> Result<T, RawRc<T, A>>
    where
        A: Allocator,
        R: RcOps,
    {
        unsafe fn inner<R>(value_ptr: NonNull<()>) -> bool
        where
            R: RcOps,
        {
            unsafe { R::lock_strong_count(strong_count_ptr_from_value_ptr(value_ptr).as_ref()) }
        }

        let is_last_strong_ref = unsafe { inner::<R>(self.weak.ptr.cast()) };

        if is_last_strong_ref {
            Ok(unsafe { self.weak.assume_init_into_inner::<R>() })
        } else {
            Err(self)
        }
    }

    pub(crate) unsafe fn unwrap_or_clone<R>(self) -> T
    where
        T: Clone,
        A: Allocator,
        R: RcOps,
    {
        unsafe {
            self.try_unwrap::<R>().unwrap_or_else(|mut rc| {
                let guard = RcGuard::<T, A, R>::new(&mut rc);

                T::clone(guard.rc.as_ref())
            })
        }
    }
}

impl<T, A> RawRc<MaybeUninit<T>, A> {
    pub(crate) fn try_new_uninit() -> Result<Self, AllocError>
    where
        A: Allocator + Default,
    {
        RawWeak::try_new_uninit::<1>().map(|weak| unsafe { Self::from_weak(weak) })
    }

    pub(crate) fn try_new_uninit_in(alloc: A) -> Result<Self, AllocError>
    where
        A: Allocator,
    {
        RawWeak::try_new_uninit_in::<1>(alloc).map(|weak| unsafe { Self::from_weak(weak) })
    }

    pub(crate) fn try_new_zeroed() -> Result<Self, AllocError>
    where
        A: Allocator + Default,
    {
        RawWeak::try_new_zeroed::<1>().map(|weak| unsafe { Self::from_weak(weak) })
    }

    pub(crate) fn try_new_zeroed_in(alloc: A) -> Result<Self, AllocError>
    where
        A: Allocator,
    {
        RawWeak::try_new_zeroed_in::<1>(alloc).map(|weak| unsafe { Self::from_weak(weak) })
    }

    #[cfg(not(no_global_oom_handling))]
    pub(crate) fn new_uninit() -> Self
    where
        A: Allocator + Default,
    {
        unsafe { Self::from_weak(RawWeak::new_uninit::<1>()) }
    }

    #[cfg(not(no_global_oom_handling))]
    pub(crate) fn new_uninit_in(alloc: A) -> Self
    where
        A: Allocator,
    {
        unsafe { Self::from_weak(RawWeak::new_uninit_in::<1>(alloc)) }
    }

    #[cfg(not(no_global_oom_handling))]
    pub(crate) fn new_zeroed() -> Self
    where
        A: Allocator + Default,
    {
        unsafe { Self::from_weak(RawWeak::new_zeroed::<1>()) }
    }

    #[cfg(not(no_global_oom_handling))]
    pub(crate) fn new_zeroed_in(alloc: A) -> Self
    where
        A: Allocator,
    {
        unsafe { Self::from_weak(RawWeak::new_zeroed_in::<1>(alloc)) }
    }

    pub(crate) unsafe fn assume_init(self) -> RawRc<T, A> {
        unsafe { self.cast() }
    }
}

impl<T, A> RawRc<[T], A> {
    #[cfg(not(no_global_oom_handling))]
    unsafe fn from_iter_exact<I>(iter: I, length: usize) -> Self
    where
        A: Allocator + Default,
        I: Iterator<Item = T>,
    {
        /// Used for dropping initialized elements in the slice if the iteration process panics.
        struct Guard<T> {
            head: NonNull<T>,
            tail: NonNull<T>,
        }

        impl<T> Drop for Guard<T> {
            fn drop(&mut self) {
                unsafe {
                    let length = self.tail.offset_from_unsigned(self.head);

                    NonNull::<[T]>::slice_from_raw_parts(self.head, length).drop_in_place();
                }
            }
        }

        let rc_layout = RcLayout::new_array::<T>(length);

        unsafe {
            let (ptr, alloc) = allocate_with::<A, _, 1>(rc_layout, |ptr| {
                let ptr = ptr.cast::<T>();
                let mut guard = Guard::<T> { head: ptr, tail: ptr };

                iter.for_each(|value| {
                    guard.tail.write(value);
                    guard.tail = guard.tail.add(1);
                });

                mem::forget(guard);
            });

            Self::from_raw_parts(NonNull::slice_from_raw_parts(ptr.cast::<T>(), length), alloc)
        }
    }

    pub(crate) unsafe fn into_array<const N: usize, R>(self) -> Option<RawRc<[T; N], A>>
    where
        A: Allocator,
        R: RcOps,
    {
        match RawRc::<[T; N], A>::try_from(self) {
            Ok(result) => Some(result),
            Err(mut raw_rc) => {
                unsafe { raw_rc.drop::<R>() };

                None
            }
        }
    }
}

impl<T, A> RawRc<[MaybeUninit<T>], A> {
    #[cfg(not(no_global_oom_handling))]
    pub(crate) fn new_uninit_slice(length: usize) -> Self
    where
        A: Allocator + Default,
    {
        unsafe { Self::from_weak(RawWeak::new_uninit_slice::<1>(length)) }
    }

    #[cfg(not(no_global_oom_handling))]
    pub(crate) fn new_uninit_slice_in(length: usize, alloc: A) -> Self
    where
        A: Allocator,
    {
        unsafe { Self::from_weak(RawWeak::new_uninit_slice_in::<1>(length, alloc)) }
    }

    #[cfg(not(no_global_oom_handling))]
    pub(crate) fn new_zeroed_slice(length: usize) -> Self
    where
        A: Allocator + Default,
    {
        unsafe { Self::from_weak(RawWeak::new_zeroed_slice::<1>(length)) }
    }

    #[cfg(not(no_global_oom_handling))]
    pub(crate) fn new_zeroed_slice_in(length: usize, alloc: A) -> Self
    where
        A: Allocator,
    {
        unsafe { Self::from_weak(RawWeak::new_zeroed_slice_in::<1>(length, alloc)) }
    }

    pub(crate) unsafe fn assume_init(self) -> RawRc<[T], A> {
        unsafe { self.cast_with(|ptr| NonNull::new_unchecked(ptr.as_ptr() as _)) }
    }
}

impl<A> RawRc<dyn Any, A> {
    pub(crate) fn downcast<T>(self) -> Result<RawRc<T, A>, Self>
    where
        T: Any,
    {
        if self.as_ref().is::<T>() { Ok(unsafe { self.downcast_unchecked() }) } else { Err(self) }
    }

    pub(crate) unsafe fn downcast_unchecked<T>(self) -> RawRc<T, A>
    where
        T: Any,
    {
        unsafe { self.cast() }
    }
}

#[cfg(not(no_sync))]
impl<A> RawRc<dyn Any + Send + Sync, A> {
    pub(crate) fn downcast<T>(self) -> Result<RawRc<T, A>, Self>
    where
        T: Any,
    {
        if self.as_ref().is::<T>() { Ok(unsafe { self.downcast_unchecked() }) } else { Err(self) }
    }

    pub(crate) unsafe fn downcast_unchecked<T>(self) -> RawRc<T, A>
    where
        T: Any,
    {
        unsafe { self.cast() }
    }
}

impl<T, A> AsRef<T> for RawRc<T, A>
where
    T: ?Sized,
{
    fn as_ref(&self) -> &T {
        unsafe { self.weak.ptr.as_ref() }
    }
}

impl<T, U, A> CoerceUnsized<RawRc<U, A>> for RawRc<T, A>
where
    T: Unsize<U> + ?Sized,
    U: ?Sized,
{
}

impl<T, A> Debug for RawRc<T, A>
where
    T: Debug + ?Sized,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        <T as Debug>::fmt(self.as_ref(), f)
    }
}

impl<T, A> Display for RawRc<T, A>
where
    T: Display + ?Sized,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        <T as Display>::fmt(self.as_ref(), f)
    }
}

impl<T, U> DispatchFromDyn<RawRc<U, Global>> for RawRc<T, Global>
where
    T: Unsize<U> + ?Sized,
    U: ?Sized,
{
}

impl<T, A> Error for RawRc<T, A>
where
    T: Error + ?Sized,
{
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        T::source(self.as_ref())
    }

    #[allow(deprecated, deprecated_in_future)]
    fn description(&self) -> &str {
        T::description(self.as_ref())
    }

    #[allow(deprecated)]
    fn cause(&self) -> Option<&dyn Error> {
        T::cause(self.as_ref())
    }

    fn provide<'a>(&'a self, request: &mut Request<'a>) {
        T::provide(self.as_ref(), request);
    }
}

impl<T, A> Pointer for RawRc<T, A>
where
    T: ?Sized,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        <&T as Pointer>::fmt(&self.as_ref(), f)
    }
}

#[cfg(not(no_global_oom_handling))]
impl<T, A> Default for RawRc<T, A>
where
    T: Default,
    A: Allocator + Default,
{
    fn default() -> Self {
        Self::new_with(T::default)
    }
}

#[cfg(not(no_global_oom_handling))]
impl<T, A> Default for RawRc<[T], A>
where
    A: Allocator + Default,
{
    fn default() -> Self {
        RawRc::<[T; 0], A>::default()
    }
}

#[cfg(not(no_global_oom_handling))]
impl<A> Default for RawRc<str, A>
where
    A: Allocator + Default,
{
    fn default() -> Self {
        let empty_slice = RawRc::<[u8], A>::default();

        // SAFETY: Empty slice is a valid `str`.
        unsafe { empty_slice.cast_with(|ptr| NonNull::new_unchecked(ptr.as_ptr() as *mut _)) }
    }
}

#[cfg(not(no_global_oom_handling))]
impl<T, A> From<T> for RawRc<T, A>
where
    A: Allocator + Default,
{
    fn from(value: T) -> Self {
        Self::new(value)
    }
}

#[cfg(not(no_global_oom_handling))]
impl<T, A> From<Box<T, A>> for RawRc<T, A>
where
    T: ?Sized,
    A: Allocator,
{
    fn from(value: Box<T, A>) -> Self {
        let value_ref = &*value;
        let alloc_ref = Box::allocator(&value);

        unsafe {
            let value_ptr = allocate_with_value_in::<T, A, 1>(value_ref, alloc_ref);
            let (box_ptr, alloc) = Box::into_raw_with_allocator(value);

            drop(Box::from_raw_in(box_ptr as *mut ManuallyDrop<T>, &alloc));

            Self::from_raw_parts(value_ptr, alloc)
        }
    }
}

#[cfg(not(no_global_oom_handling))]
trait SpecRawRcFromSlice<T> {
    fn spec_from_slice(slice: &[T]) -> Self;
}

#[cfg(not(no_global_oom_handling))]
impl<T, A> SpecRawRcFromSlice<T> for RawRc<[T], A>
where
    T: Clone,
    A: Allocator + Default,
{
    default fn spec_from_slice(slice: &[T]) -> Self {
        unsafe { Self::from_iter_exact(slice.iter().cloned(), slice.len()) }
    }
}

#[cfg(not(no_global_oom_handling))]
impl<T, A> SpecRawRcFromSlice<T> for RawRc<[T], A>
where
    T: Copy,
    A: Allocator + Default,
{
    fn spec_from_slice(slice: &[T]) -> Self {
        let (ptr, alloc) = allocate_with_value::<[T], A, 1>(slice);

        unsafe { Self::from_raw_parts(ptr, alloc) }
    }
}

#[cfg(not(no_global_oom_handling))]
impl<T, A> From<&[T]> for RawRc<[T], A>
where
    T: Clone,
    A: Allocator + Default,
{
    fn from(value: &[T]) -> Self {
        Self::spec_from_slice(value)
    }
}

#[cfg(not(no_global_oom_handling))]
impl<T, A> From<&mut [T]> for RawRc<[T], A>
where
    T: Clone,
    A: Allocator + Default,
{
    fn from(value: &mut [T]) -> Self {
        Self::from(&*value)
    }
}

#[cfg(not(no_global_oom_handling))]
impl<A> From<&str> for RawRc<str, A>
where
    A: Allocator + Default,
{
    #[inline]
    fn from(value: &str) -> Self {
        let rc_of_bytes = RawRc::<[u8], A>::from(value.as_bytes());

        unsafe { rc_of_bytes.cast_with(|ptr| NonNull::new_unchecked(ptr.as_ptr() as _)) }
    }
}

#[cfg(not(no_global_oom_handling))]
impl<A> From<&mut str> for RawRc<str, A>
where
    A: Allocator + Default,
{
    fn from(value: &mut str) -> Self {
        Self::from(&*value)
    }
}

#[cfg(not(no_global_oom_handling))]
impl From<String> for RawRc<str, Global> {
    fn from(value: String) -> Self {
        let rc_of_bytes = RawRc::<[u8], Global>::from(value.into_bytes());

        unsafe { rc_of_bytes.cast_with(|ptr| NonNull::new_unchecked(ptr.as_ptr() as _)) }
    }
}

impl<A> From<RawRc<str, A>> for RawRc<[u8], A> {
    fn from(value: RawRc<str, A>) -> Self {
        unsafe { value.cast_with(|ptr| NonNull::new_unchecked(ptr.as_ptr() as _)) }
    }
}

#[cfg(not(no_global_oom_handling))]
impl<T, const N: usize, A> From<[T; N]> for RawRc<[T], A>
where
    A: Allocator + Default,
{
    fn from(value: [T; N]) -> Self {
        RawRc::new(value)
    }
}

#[cfg(not(no_global_oom_handling))]
impl<T, A> From<Vec<T, A>> for RawRc<[T], A>
where
    A: Allocator,
{
    fn from(value: Vec<T, A>) -> Self {
        let src = &*value;
        let alloc = value.allocator();
        let value_ptr = allocate_with_value_in::<[T], A, 1>(src, alloc);
        let (vec_ptr, _length, capacity, alloc) = value.into_raw_parts_with_alloc();

        unsafe {
            drop(Vec::from_raw_parts_in(vec_ptr, 0, capacity, &alloc));

            Self::from_raw_parts(value_ptr, alloc)
        }
    }
}

impl<T, const N: usize, A> TryFrom<RawRc<[T], A>> for RawRc<[T; N], A> {
    type Error = RawRc<[T], A>;

    fn try_from(value: RawRc<[T], A>) -> Result<Self, Self::Error> {
        if value.as_ref().len() == N { Ok(unsafe { value.cast() }) } else { Err(value) }
    }
}

#[cfg(not(no_global_oom_handling))]
trait SpecRawRcFromIter<I> {
    fn spec_from_iter(iter: I) -> Self;
}

#[cfg(not(no_global_oom_handling))]
impl<I> SpecRawRcFromIter<I> for RawRc<[I::Item], Global>
where
    I: Iterator,
{
    default fn spec_from_iter(iter: I) -> Self {
        Self::from(iter.collect::<Vec<_>>())
    }
}

#[cfg(not(no_global_oom_handling))]
impl<I> SpecRawRcFromIter<I> for RawRc<[I::Item], Global>
where
    I: TrustedLen,
{
    fn spec_from_iter(iter: I) -> Self {
        // This is the case for a `TrustedLen` iterator.

        if let (low, Some(high)) = iter.size_hint() {
            debug_assert_eq!(
                low,
                high,
                "TrustedLen iterator's size hint is not exact: {:?}",
                (low, high)
            );

            // SAFETY: We need to ensure that the iterator has an exact length and we have.
            unsafe { Self::from_iter_exact(iter, low) }
        } else {
            // TrustedLen contract guarantees that `upper_bound == None` implies an iterator
            // length exceeding `usize::MAX`.
            // The default implementation would collect into a vec which would panic.
            // Thus we panic here immediately without invoking `Vec` code.
            panic!("capacity overflow");
        }
    }
}

#[cfg(not(no_global_oom_handling))]
impl<T> FromIterator<T> for RawRc<[T], Global> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        Self::spec_from_iter(iter.into_iter())
    }
}

impl<T, A> Hash for RawRc<T, A>
where
    T: Hash + ?Sized,
{
    fn hash<H: Hasher>(&self, state: &mut H) {
        T::hash(self.as_ref(), state);
    }
}

// Hack to allow specializing on `Eq` even though `Eq` has a method.
#[rustc_unsafe_specialization_marker]
trait MarkerEq: PartialEq<Self> {}

impl<T> MarkerEq for T where T: Eq {}

trait SpecPartialEq {
    fn spec_eq(&self, other: &Self) -> bool;
    fn spec_ne(&self, other: &Self) -> bool;
}

impl<T, A> SpecPartialEq for RawRc<T, A>
where
    T: PartialEq + ?Sized,
{
    #[inline]
    default fn spec_eq(&self, other: &Self) -> bool {
        T::eq(self.as_ref(), other.as_ref())
    }

    #[inline]
    default fn spec_ne(&self, other: &Self) -> bool {
        T::ne(self.as_ref(), other.as_ref())
    }
}

impl<T, A> SpecPartialEq for RawRc<T, A>
where
    T: MarkerEq + ?Sized,
{
    #[inline]
    fn spec_eq(&self, other: &Self) -> bool {
        Self::ptr_eq(self, other) || T::eq(self.as_ref(), other.as_ref())
    }

    #[inline]
    fn spec_ne(&self, other: &Self) -> bool {
        Self::ptr_ne(self, other) && T::ne(self.as_ref(), other.as_ref())
    }
}

impl<T, A> PartialEq for RawRc<T, A>
where
    T: PartialEq + ?Sized,
{
    fn eq(&self, other: &Self) -> bool {
        Self::spec_eq(self, other)
    }

    fn ne(&self, other: &Self) -> bool {
        Self::spec_ne(self, other)
    }
}

impl<T, A> Eq for RawRc<T, A> where T: Eq + ?Sized {}

impl<T, A> PartialOrd for RawRc<T, A>
where
    T: PartialOrd + ?Sized,
{
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        T::partial_cmp(self.as_ref(), other.as_ref())
    }

    fn lt(&self, other: &Self) -> bool {
        T::lt(self.as_ref(), other.as_ref())
    }

    fn le(&self, other: &Self) -> bool {
        T::le(self.as_ref(), other.as_ref())
    }

    fn gt(&self, other: &Self) -> bool {
        T::gt(self.as_ref(), other.as_ref())
    }

    fn ge(&self, other: &Self) -> bool {
        T::ge(self.as_ref(), other.as_ref())
    }
}

impl<T, A> Ord for RawRc<T, A>
where
    T: Ord + ?Sized,
{
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        T::cmp(self.as_ref(), other.as_ref())
    }
}

unsafe impl<T, A> PinCoerceUnsized for RawRc<T, A>
where
    T: ?Sized,
    A: Allocator,
{
}

/// A uniquely owned `RawRc` that allows multiple weak references but only one strong reference.
/// `RawUniqueRc` does not implement `Drop`, user should call `RawUniqueRc::drop` manually to drop
/// this object.
#[repr(transparent)]
pub(crate) struct RawUniqueRc<T, A>
where
    T: ?Sized,
{
    // A `RawUniqueRc` is just a `RawWeak` that has zero strong count but with the value
    // initialized.
    weak: RawWeak<T, A>,

    // Defines the ownership of `T` for drop-check.
    _marker: PhantomData<T>,

    // Invariance is necessary for soundness: once other `RawWeak` references exist, we already have
    // a form of shared mutability!
    _marker2: PhantomData<*mut T>,
}

impl<T, A> RawUniqueRc<T, A>
where
    T: ?Sized,
{
    pub(crate) unsafe fn downgrade<R>(&self) -> RawWeak<T, A>
    where
        A: Clone,
        R: RcOps,
    {
        unsafe { self.weak.clone::<R>() }
    }

    pub(crate) unsafe fn drop<R>(&mut self)
    where
        A: Allocator,
        R: RcOps,
    {
        unsafe { self.weak.assume_init_drop::<R>() };
    }

    pub(crate) unsafe fn into_rc<R>(self) -> RawRc<T, A>
    where
        R: RcOps,
    {
        unsafe fn inner<R>(value_ptr: NonNull<()>)
        where
            R: RcOps,
        {
            unsafe { R::unlock_strong_count(strong_count_ptr_from_value_ptr(value_ptr).as_ref()) };
        }

        unsafe {
            inner::<R>(self.weak.ptr.cast());

            RawRc::from_weak(self.weak)
        }
    }
}

impl<T, A> RawUniqueRc<T, A> {
    #[cfg(not(no_global_oom_handling))]
    unsafe fn from_weak_with_value(weak: RawWeak<T, A>, value: T) -> Self {
        unsafe { weak.ptr.write(value) };

        Self { weak, _marker: PhantomData, _marker2: PhantomData }
    }

    #[cfg(not(no_global_oom_handling))]
    pub(crate) fn new(value: T) -> Self
    where
        A: Allocator + Default,
    {
        unsafe { Self::from_weak_with_value(RawWeak::new_uninit::<0>(), value) }
    }

    #[cfg(not(no_global_oom_handling))]
    pub(crate) fn new_in(value: T, alloc: A) -> Self
    where
        A: Allocator,
    {
        unsafe { Self::from_weak_with_value(RawWeak::new_uninit_in::<0>(alloc), value) }
    }
}

impl<T, A> AsMut<T> for RawUniqueRc<T, A>
where
    T: ?Sized,
{
    fn as_mut(&mut self) -> &mut T {
        unsafe { self.weak.ptr.as_mut() }
    }
}

impl<T, A> AsRef<T> for RawUniqueRc<T, A>
where
    T: ?Sized,
{
    fn as_ref(&self) -> &T {
        unsafe { self.weak.ptr.as_ref() }
    }
}

impl<T, U, A> CoerceUnsized<RawUniqueRc<U, A>> for RawUniqueRc<T, A>
where
    T: Unsize<U> + ?Sized,
    U: ?Sized,
    A: Allocator,
{
}

impl<T, A> Debug for RawUniqueRc<T, A>
where
    T: Debug + ?Sized,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        <T as Debug>::fmt(self.as_ref(), f)
    }
}

impl<T, U> DispatchFromDyn<RawUniqueRc<U, Global>> for RawUniqueRc<T, Global>
where
    T: Unsize<U> + ?Sized,
    U: ?Sized,
{
}

impl<T, A> Display for RawUniqueRc<T, A>
where
    T: Display + ?Sized,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        <T as Display>::fmt(self.as_ref(), f)
    }
}

impl<T, A> Eq for RawUniqueRc<T, A> where T: Eq + ?Sized {}

impl<T, A> Hash for RawUniqueRc<T, A>
where
    T: Hash + ?Sized,
{
    fn hash<H: Hasher>(&self, state: &mut H) {
        T::hash(self.as_ref(), state);
    }
}

impl<T, A> Ord for RawUniqueRc<T, A>
where
    T: Ord + ?Sized,
{
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        T::cmp(self.as_ref(), other.as_ref())
    }
}

impl<T, A> PartialEq for RawUniqueRc<T, A>
where
    T: PartialEq + ?Sized,
{
    fn eq(&self, other: &Self) -> bool {
        T::eq(self.as_ref(), other.as_ref())
    }

    fn ne(&self, other: &Self) -> bool {
        T::ne(self.as_ref(), other.as_ref())
    }
}

impl<T, A> PartialOrd for RawUniqueRc<T, A>
where
    T: PartialOrd + ?Sized,
{
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        T::partial_cmp(self.as_ref(), other.as_ref())
    }

    fn lt(&self, other: &Self) -> bool {
        T::lt(self.as_ref(), other.as_ref())
    }

    fn le(&self, other: &Self) -> bool {
        T::le(self.as_ref(), other.as_ref())
    }

    fn gt(&self, other: &Self) -> bool {
        T::gt(self.as_ref(), other.as_ref())
    }

    fn ge(&self, other: &Self) -> bool {
        T::ge(self.as_ref(), other.as_ref())
    }
}

impl<T, A> Pointer for RawUniqueRc<T, A>
where
    T: ?Sized,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        <&T as Pointer>::fmt(&self.as_ref(), f)
    }
}
