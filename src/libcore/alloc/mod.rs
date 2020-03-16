//! Memory allocation APIs

#![stable(feature = "alloc_module", since = "1.28.0")]

mod global;
mod layout;

#[stable(feature = "global_alloc", since = "1.28.0")]
pub use self::global::GlobalAlloc;
#[stable(feature = "alloc_layout", since = "1.28.0")]
pub use self::layout::{Layout, LayoutErr};

use crate::fmt;
use crate::ptr::{self, NonNull};

/// The `AllocErr` error indicates an allocation failure
/// that may be due to resource exhaustion or to
/// something wrong when combining the given input arguments with this
/// allocator.
#[unstable(feature = "allocator_api", issue = "32838")]
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct AllocErr;

// (we need this for downstream impl of trait Error)
#[unstable(feature = "allocator_api", issue = "32838")]
impl fmt::Display for AllocErr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("memory allocation failed")
    }
}

/// A desired initial state for allocated memory.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
#[unstable(feature = "allocator_api", issue = "32838")]
pub enum AllocInit {
    /// The contents of the new memory are undefined.
    ///
    /// Reading uninitialized memory is Undefined Behavior; it must be initialized before use.
    Uninitialized,
    /// The new memory is guaranteed to be zeroed.
    Zeroed,
}

impl AllocInit {
    /// Initialize the memory block referenced by `ptr` and specified by `Layout`.
    ///
    /// This behaves like calling [`AllocInit::initialize_offset(ptr, layout, 0)`][off].
    ///
    /// [off]: AllocInit::initialize_offset
    ///
    /// # Safety
    ///
    /// * `layout` must [*fit*] the block of memory referenced by `ptr`
    ///
    /// [*fit*]: trait.AllocRef.html#memory-fitting
    #[inline]
    #[unstable(feature = "allocator_api", issue = "32838")]
    pub unsafe fn initialize(self, ptr: NonNull<u8>, layout: Layout) {
        self.initialize_offset(ptr, layout, 0)
    }

    /// Initialize the memory block referenced by `ptr` and specified by `Layout` at the specified
    /// `offset`.
    ///
    /// This is a no-op for [`AllocInit::Uninitialized`] and writes zeroes for [`AllocInit::Zeroed`]
    /// at `ptr + offset` until `ptr + layout.size()`.
    ///
    /// # Safety
    ///
    /// * `layout` must [*fit*] the block of memory referenced by `ptr`
    ///
    /// * `offset` must be smaller than or equal to `layout.size()`
    ///
    /// [*fit*]: trait.AllocRef.html#memory-fitting
    #[unstable(feature = "allocator_api", issue = "32838")]
    pub unsafe fn initialize_offset(self, ptr: NonNull<u8>, layout: Layout, offset: usize) {
        debug_assert!(
            offset <= layout.size(),
            "`offset` must be smaller than or equal to `layout.size()`"
        );
        match self {
            AllocInit::Uninitialized => (),
            AllocInit::Zeroed => {
                let new_ptr = ptr.as_ptr().add(offset);
                let size = layout.size() - offset;
                ptr::write_bytes(new_ptr, 0, size);
            }
        }
    }
}

/// A placement constraint when growing or shrinking an existing allocation.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
#[unstable(feature = "allocator_api", issue = "32838")]
pub enum ReallocPlacement {
    /// The allocator is allowed to move the allocation to a different memory address.
    // FIXME(wg-allocators#46): Add a section to the module documentation "What is a legal
    //                          allocator" and link it at "valid location".
    ///
    /// If the allocation _does_ move, it's the responsibility of the allocator
    /// to also move the data from the previous location to the new location.
    MayMove,
    /// The address of the new memory must not change.
    ///
    /// If the allocation would have to be moved to a new location to fit, the
    /// reallocation request will fail.
    InPlace,
}

/// An implementation of `AllocRef` can allocate, grow, shrink, and deallocate arbitrary blocks of
/// data described via [`Layout`][].
///
/// `AllocRef` is designed to be implemented on ZSTs, references, or smart pointers because having
/// an allocator like `MyAlloc([u8; N])` cannot be moved, without updating the pointers to the
/// allocated memory.
///
/// Unlike [`GlobalAlloc`][], zero-sized allocations are allowed in `AllocRef`. If an underlying
/// allocator does not support this (like jemalloc) or return a null pointer (such as
/// `libc::malloc`), this case must be caught. [`Layout::dangling()`][] then can be used to create
/// an aligned `NonNull<u8>`.
///
/// ### Currently allocated memory
///
/// Some of the methods require that a memory block be *currently allocated* via an allocator. This
/// means that:
///
/// * the starting address for that memory block was previously returned by [`alloc`], [`grow`], or
///   [`shrink`], and
///
/// * the memory block has not been subsequently deallocated, where blocks are either deallocated
///   directly by being passed to [`dealloc`] or were changed by being passed to [`grow`] or
///   [`shrink`] that returns `Ok`. If `grow` or `shrink` have returned `Err`, the passed pointer
///   remains valid.
///
/// [`alloc`]: AllocRef::alloc
/// [`grow`]: AllocRef::grow
/// [`shrink`]: AllocRef::shrink
/// [`dealloc`]: AllocRef::dealloc
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
///   - `max` is the latest actual size returned from [`alloc`], [`grow`], or [`shrink`].
///
/// [`layout.align()`]: Layout::align
/// [`layout.size()`]: Layout::size
///
/// ### Notes
///
///  * if a layout `k` fits a memory block (denoted by `ptr`) currently allocated via an allocator
///    `a`, then it is legal to use that layout to deallocate it, i.e.,
///    [`a.dealloc(ptr, k);`][`dealloc`], and
///
///  * if an allocator does not support overallocating, it is fine to simply return
///    [`layout.size()`] as the actual size.
///
/// # Safety
///
/// * Pointers returned from an allocator must point to valid memory and retain their validity until
///   the instance and all of its clones are dropped,
///
/// * cloning or moving the allocator must not invalidate pointers returned from this allocator.
///   A cloned allocator must behave like the same allocator, and
///
/// * any pointer to a memory block which is [*currently allocated*] may be passed to any other
///   method of the allocator.
///
/// [*currently allocated*]: #currently-allocated-memory
#[unstable(feature = "allocator_api", issue = "32838")]
pub unsafe trait AllocRef {
    /// On success, returns a pointer meeting the size and alignment guarantees of `layout` and the
    /// actual size of the allocated block, which is greater than or equal to `layout.size()`.
    ///
    /// The returned block of storage is initialized as specified by [`init`], all the way up to
    /// the returned `actual_size`.
    ///
    /// [`init`]: AllocInit
    ///
    /// # Errors
    ///
    /// Returning `Err` indicates that either memory is exhausted or `layout` does not meet
    /// allocator's size or alignment constraints.
    ///
    /// Implementations are encouraged to return `Err` on memory exhaustion rather than panicking or
    /// aborting, but this is not a strict requirement. (Specifically: it is *legal* to implement
    /// this trait atop an underlying native allocation library that aborts on memory exhaustion.)
    ///
    /// Clients wishing to abort computation in response to an allocation error are encouraged to
    /// call the [`handle_alloc_error`] function, rather than directly invoking `panic!` or similar.
    ///
    /// [`handle_alloc_error`]: ../../alloc/alloc/fn.handle_alloc_error.html
    fn alloc(&mut self, layout: Layout, init: AllocInit) -> Result<(NonNull<u8>, usize), AllocErr>;

    /// Deallocates the memory referenced by `ptr`.
    ///
    /// # Safety
    ///
    /// * `ptr` must denote a block of memory [*currently allocated*] via this allocator,
    ///
    /// * `layout` must [*fit*] that block of memory, and
    ///
    /// * the alignment of the `layout` must match the alignment used to allocate that block of
    ///   memory.
    ///
    /// [*currently allocated*]: #currently-allocated-memory
    /// [*fit*]: #memory-fitting
    unsafe fn dealloc(&mut self, ptr: NonNull<u8>, layout: Layout);

    /// Attempts to extend the allocation referenced by `ptr` to fit `new_size`.
    ///
    /// Returns a pointer and the actual size of the allocated block. The pointer is suitable for
    /// holding data described by a new layout with `layout`’s alignment and a size given by
    /// `new_size`. To accomplish this, the allocator may extend the allocation referenced by `ptr`
    /// to fit the new layout.
    ///
    /// If this returns `Ok`, then ownership of the memory block referenced by `ptr` has been
    /// transferred to this allocator. The memory may or may not have been freed, and should be
    /// considered unusable (unless of course it was transferred back to the caller again via the
    /// return value of this method).
    ///
    /// If this method returns `Err`, then ownership of the memory block has not been transferred to
    /// this allocator, and the contents of the memory block are unaltered.
    ///
    /// The behavior of how the allocator tries to grow the memory is specified by [`placement`].
    /// The first `layout.size()` bytes of memory are preserved or copied as appropriate from `ptr`,
    /// and the remaining bytes, from `layout.size()` to the returned actual size, are initialized
    /// according to [`init`].
    ///
    /// [`placement`]: ReallocPlacement
    /// [`init`]: AllocInit
    ///
    /// # Safety
    ///
    /// * `ptr` must be [*currently allocated*] via this allocator,
    ///
    /// * `layout` must [*fit*] the `ptr`. (The `new_size` argument need not fit it.)
    ///
    // We can't require that `new_size` is strictly greater than `layout.size()` because of ZSTs.
    // An alternative would be
    // * `new_size must be strictly greater than `layout.size()` or both are zero
    /// * `new_size` must be greater than or equal to `layout.size()`
    ///
    /// * `new_size`, when rounded up to the nearest multiple of `layout.align()`, must not overflow
    ///   (i.e., the rounded value must be less than `usize::MAX`).
    ///
    /// [*currently allocated*]: #currently-allocated-memory
    /// [*fit*]: #memory-fitting
    ///
    /// # Errors
    ///
    /// Returns `Err` if the new layout does not meet the allocator's size and alignment
    /// constraints of the allocator, or if growing otherwise fails.
    ///
    /// Implementations are encouraged to return `Err` on memory exhaustion rather than panicking or
    /// aborting, but this is not a strict requirement. (Specifically: it is *legal* to implement
    /// this trait atop an underlying native allocation library that aborts on memory exhaustion.)
    ///
    /// Clients wishing to abort computation in response to an allocation error are encouraged to
    /// call the [`handle_alloc_error`] function, rather than directly invoking `panic!` or similar.
    ///
    /// [`handle_alloc_error`]: ../../alloc/alloc/fn.handle_alloc_error.html
    unsafe fn grow(
        &mut self,
        ptr: NonNull<u8>,
        layout: Layout,
        new_size: usize,
        placement: ReallocPlacement,
        init: AllocInit,
    ) -> Result<(NonNull<u8>, usize), AllocErr> {
        let old_size = layout.size();
        debug_assert!(
            new_size >= old_size,
            "`new_size` must be greater than or equal to `layout.size()`"
        );

        if new_size == old_size {
            return Ok((ptr, new_size));
        }

        match placement {
            ReallocPlacement::MayMove => {
                let (new_ptr, alloc_size) =
                    self.alloc(Layout::from_size_align_unchecked(new_size, layout.align()), init)?;
                ptr::copy_nonoverlapping(ptr.as_ptr(), new_ptr.as_ptr(), old_size);
                self.dealloc(ptr, layout);
                Ok((new_ptr, alloc_size))
            }
            ReallocPlacement::InPlace => Err(AllocErr),
        }
    }

    /// Attempts to shrink the allocation referenced by `ptr` to fit `new_size`.
    ///
    /// Returns a pointer and the actual size of the allocated block. The pointer is suitable for
    /// holding data described by a new layout with `layout`’s alignment and a size given by
    /// `new_size`. To accomplish this, the allocator may shrink the allocation referenced by `ptr`
    /// to fit the new layout.
    ///
    /// The behavior on how the allocator tries to shrink the memory can be specified by
    /// [`placement`].
    ///
    /// If this returns `Ok`, then ownership of the memory block referenced by `ptr` has been
    /// transferred to this allocator. The memory may or may not have been freed, and should be
    /// considered unusable unless it was transferred back to the caller again via the
    /// return value of this method.
    ///
    /// If this method returns `Err`, then ownership of the memory block has not been transferred to
    /// this allocator, and the contents of the memory block are unaltered.
    ///
    /// [`placement`]: ReallocPlacement
    ///
    /// # Safety
    ///
    /// * `ptr` must be [*currently allocated*] via this allocator,
    ///
    /// * `layout` must [*fit*] the `ptr`. (The `new_size` argument need not fit it.)
    ///
    // We can't require that `new_size` is strictly smaller than `layout.size()` because of ZSTs.
    // An alternative would be
    // * `new_size must be strictly smaller than `layout.size()` or both are zero
    /// * `new_size` must be smaller than or equal to `layout.size()`
    ///
    /// [*currently allocated*]: #currently-allocated-memory
    /// [*fit*]: #memory-fitting
    ///
    /// # Errors
    ///
    /// Returns `Err` if the new layout does not meet the allocator's size and alignment
    /// constraints of the allocator, or if shrinking otherwise fails.
    ///
    /// Implementations are encouraged to return `Err` on memory exhaustion rather than panicking or
    /// aborting, but this is not a strict requirement. (Specifically: it is *legal* to implement
    /// this trait atop an underlying native allocation library that aborts on memory exhaustion.)
    ///
    /// Clients wishing to abort computation in response to an allocation error are encouraged to
    /// call the [`handle_alloc_error`] function, rather than directly invoking `panic!` or similar.
    ///
    /// [`handle_alloc_error`]: ../../alloc/alloc/fn.handle_alloc_error.html
    unsafe fn shrink(
        &mut self,
        ptr: NonNull<u8>,
        layout: Layout,
        new_size: usize,
        placement: ReallocPlacement,
    ) -> Result<(NonNull<u8>, usize), AllocErr> {
        let old_size = layout.size();
        debug_assert!(
            new_size <= old_size,
            "`new_size` must be smaller than or equal to `layout.size()`"
        );

        if new_size == old_size {
            return Ok((ptr, new_size));
        }

        match placement {
            ReallocPlacement::MayMove => {
                let (new_ptr, alloc_size) = self.alloc(
                    Layout::from_size_align_unchecked(new_size, layout.align()),
                    AllocInit::Uninitialized,
                )?;
                ptr::copy_nonoverlapping(ptr.as_ptr(), new_ptr.as_ptr(), new_size);
                self.dealloc(ptr, layout);
                Ok((new_ptr, alloc_size))
            }
            ReallocPlacement::InPlace => Err(AllocErr),
        }
    }
}
