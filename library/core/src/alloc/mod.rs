//! Memory allocation APIs

#![stable(feature = "alloc_module", since = "1.28.0")]

mod global;
mod layout;

#[stable(feature = "global_alloc", since = "1.28.0")]
pub use self::global::GlobalAlloc;
#[stable(feature = "alloc_layout", since = "1.28.0")]
pub use self::layout::Layout;
#[stable(feature = "alloc_layout", since = "1.28.0")]
#[deprecated(
    since = "1.52.0",
    note = "Name does not follow std convention, use LayoutError",
    suggestion = "LayoutError"
)]
#[allow(deprecated, deprecated_in_future)]
pub use self::layout::LayoutErr;
#[stable(feature = "alloc_layout_error", since = "1.50.0")]
pub use self::layout::LayoutError;
use crate::error::Error;
use crate::fmt;
use crate::ptr::{self, NonNull};

/// The `AllocError` error indicates an allocation failure
/// that may be due to resource exhaustion or to
/// something wrong when combining the given input arguments with this
/// allocator.
#[unstable(feature = "allocator_api", issue = "32838")]
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub struct AllocError;

#[unstable(
    feature = "allocator_api",
    reason = "the precise API and guarantees it provides may be tweaked.",
    issue = "32838"
)]
impl Error for AllocError {}

// (we need this for downstream impl of trait Error)
#[unstable(feature = "allocator_api", issue = "32838")]
impl fmt::Display for AllocError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("memory allocation failed")
    }
}

/// An implementation of `Allocator` can allocate, grow, shrink, and deallocate arbitrary blocks of
/// data described via [`Layout`][].
///
/// `Allocator` is designed to be implemented on ZSTs, references, or smart pointers.
/// An allocator for `MyAlloc([u8; N])` cannot be moved, without updating the pointers to the
/// allocated memory.
///
/// In contrast to [`GlobalAlloc`][], `Allocator` allows zero-sized allocations. If an underlying
/// allocator does not support this (like jemalloc) or responds by returning a null pointer
/// (such as `libc::malloc`), this must be caught by the implementation.
///
/// ### Currently allocated memory
///
/// Some of the methods require that a memory block is *currently allocated* by an allocator.
/// This means that:
///  * the starting address for that memory block was previously
///    returned by [`allocate`], [`grow`], or [`shrink`], and
///  * the memory block has not subsequently been deallocated.
///
/// A memory block is deallocated by a call to [`deallocate`],
/// or by a call to [`grow`] or [`shrink`] that returns `Ok`.
/// A call to `grow` or `shrink` that returns `Err`,
/// does not deallocate the memory block passed to it.
///
/// [`allocate`]: Allocator::allocate
/// [`grow`]: Allocator::grow
/// [`shrink`]: Allocator::shrink
/// [`deallocate`]: Allocator::deallocate
///
/// ### Memory fitting
///
/// Some of the methods require that a `layout` *fit* a memory block or vice versa. This means that the
/// following conditions must hold:
///  * the memory block must be *currently allocated* with alignment of [`layout.align()`], and
///  * [`layout.size()`] must fall in the range `min ..= max`, where:
///    - `min` is the size of the layout used to allocate the block, and
///    - `max` is the actual size returned from [`allocate`], [`grow`], or [`shrink`].
///
/// [`layout.align()`]: Layout::align
/// [`layout.size()`]: Layout::size
///
/// # Safety
///
/// Memory blocks that are [*currently allocated*] by an allocator,
/// must point to valid memory, and retain their validity until either:
///  - the memory block is deallocated, or
///  - the allocator is dropped.
///
/// Copying, cloning, or moving the allocator must not invalidate memory blocks returned from it.
/// A copied or cloned allocator must behave like the original allocator.
///
/// A memory block which is [*currently allocated*] may be passed to
/// any method of the allocator that accepts such an argument.
///
/// Additionally, any memory block returned by the allocator must
/// satisfy the allocation invariants described in `core::ptr`.
/// In particular, if a block has base address `p` and size `n`,
/// then `p as usize + n <= usize::MAX` must hold.
///
/// This ensures that pointer arithmetic within the allocation
/// (for example, `ptr.add(len)`) cannot overflow the address space.
/// [*currently allocated*]: #currently-allocated-memory
#[unstable(feature = "allocator_api", issue = "32838")]
#[rustc_const_unstable(feature = "const_heap", issue = "79597")]
pub const unsafe trait Allocator {
    /// Attempts to allocate a block of memory.
    ///
    /// On success, returns a [`NonNull<[u8]>`][NonNull] meeting the size and alignment guarantees of `layout`.
    ///
    /// The returned block may have a larger size than specified by `layout.size()`, and may or may
    /// not have its contents initialized.
    ///
    /// The returned block of memory remains valid as long as it is [*currently allocated*] and the shorter of:
    ///   - the borrow-checker lifetime of the allocator type itself.
    ///   - as long as the allocator and all its clones have not been dropped.
    ///
    /// [*currently allocated*]: #currently-allocated-memory
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
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError>;

    /// Behaves like `allocate`, but also ensures that the returned memory is zero-initialized.
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
    fn allocate_zeroed(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        let ptr = self.allocate(layout)?;
        // SAFETY: `alloc` returns a valid memory block
        unsafe { ptr.as_non_null_ptr().as_ptr().write_bytes(0, ptr.len()) }
        Ok(ptr)
    }

    /// Deallocates the memory referenced by `ptr`.
    ///
    /// # Safety
    ///
    /// * `ptr` must denote a block of memory [*currently allocated*] via this allocator, and
    /// * `layout` must [*fit*] that block of memory.
    ///
    /// [*currently allocated*]: #currently-allocated-memory
    /// [*fit*]: #memory-fitting
    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout);

    /// Attempts to extend the memory block.
    ///
    /// Returns a new [`NonNull<[u8]>`][NonNull] containing a pointer and the actual size of the allocated
    /// memory. The pointer is suitable for holding data described by `new_layout`. To accomplish
    /// this, the allocator may extend the allocation referenced by `ptr` to fit the new layout.
    ///
    /// If this returns `Ok`, then ownership of the memory block referenced by `ptr` has been
    /// transferred to this allocator. Any access to the old `ptr` is Undefined Behavior, even if the
    /// allocation was grown in-place. The newly returned pointer is the only valid pointer
    /// for accessing this memory now.
    ///
    /// If this method returns `Err`, then ownership of the memory block has not been transferred to
    /// this allocator, and the contents of the memory block are unaltered.
    ///
    /// # Safety
    ///
    /// * `ptr` must denote a block of memory [*currently allocated*] via this allocator.
    /// * `old_layout` must [*fit*] that block of memory (The `new_layout` argument need not fit it.).
    /// * `new_layout.size()` must be greater than or equal to `old_layout.size()`.
    ///
    /// Note that `new_layout.align()` need not be the same as `old_layout.align()`.
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
        &self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        debug_assert!(
            new_layout.size() >= old_layout.size(),
            "`new_layout.size()` must be greater than or equal to `old_layout.size()`"
        );

        let new_ptr = self.allocate(new_layout)?;

        // SAFETY: because `new_layout.size()` must be greater than or equal to
        // `old_layout.size()`, both the old and new memory allocation are valid for reads and
        // writes for `old_layout.size()` bytes. Also, because the old allocation wasn't yet
        // deallocated, it cannot overlap `new_ptr`. Thus, the call to `copy_nonoverlapping` is
        // safe. The safety contract for `dealloc` must be upheld by the caller.
        unsafe {
            ptr::copy_nonoverlapping(ptr.as_ptr(), new_ptr.as_mut_ptr(), old_layout.size());
            self.deallocate(ptr, old_layout);
        }

        Ok(new_ptr)
    }

    /// Behaves like `grow`, but also ensures that the new contents are set to zero before being
    /// returned.
    ///
    /// The memory block will contain the following contents after a successful call to
    /// `grow_zeroed`:
    ///   * Bytes `0..old_layout.size()` are preserved from the original allocation.
    ///   * Bytes `old_layout.size()..old_size` will either be preserved or zeroed, depending on
    ///     the allocator implementation. `old_size` refers to the size of the memory block prior
    ///     to the `grow_zeroed` call, which may be larger than the size that was originally
    ///     requested when it was allocated.
    ///   * Bytes `old_size..new_size` are zeroed. `new_size` refers to the size of the memory
    ///     block returned by the `grow_zeroed` call.
    ///
    /// # Safety
    ///
    /// * `ptr` must denote a block of memory [*currently allocated*] via this allocator.
    /// * `old_layout` must [*fit*] that block of memory (The `new_layout` argument need not fit it.).
    /// * `new_layout.size()` must be greater than or equal to `old_layout.size()`.
    ///
    /// Note that `new_layout.align()` need not be the same as `old_layout.align()`.
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
    unsafe fn grow_zeroed(
        &self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        debug_assert!(
            new_layout.size() >= old_layout.size(),
            "`new_layout.size()` must be greater than or equal to `old_layout.size()`"
        );

        let new_ptr = self.allocate_zeroed(new_layout)?;

        // SAFETY: because `new_layout.size()` must be greater than or equal to
        // `old_layout.size()`, both the old and new memory allocation are valid for reads and
        // writes for `old_layout.size()` bytes. Also, because the old allocation wasn't yet
        // deallocated, it cannot overlap `new_ptr`. Thus, the call to `copy_nonoverlapping` is
        // safe. The safety contract for `dealloc` must be upheld by the caller.
        unsafe {
            ptr::copy_nonoverlapping(ptr.as_ptr(), new_ptr.as_mut_ptr(), old_layout.size());
            self.deallocate(ptr, old_layout);
        }

        Ok(new_ptr)
    }

    /// Attempts to shrink the memory block.
    ///
    /// Returns a new [`NonNull<[u8]>`][NonNull] containing a pointer and the actual size of the allocated
    /// memory. The pointer is suitable for holding data described by `new_layout`. To accomplish
    /// this, the allocator may shrink the allocation referenced by `ptr` to fit the new layout.
    ///
    /// If this returns `Ok`, then ownership of the memory block referenced by `ptr` has been
    /// transferred to this allocator. Any access to the old `ptr` is Undefined Behavior, even if the
    /// allocation was shrunk in-place. The newly returned pointer is the only valid pointer
    /// for accessing this memory now.
    ///
    /// If this method returns `Err`, then ownership of the memory block has not been transferred to
    /// this allocator, and the contents of the memory block are unaltered.
    ///
    /// # Safety
    ///
    /// * `ptr` must denote a block of memory [*currently allocated*] via this allocator.
    /// * `old_layout` must [*fit*] that block of memory (The `new_layout` argument need not fit it.).
    /// * `new_layout.size()` must be smaller than or equal to `old_layout.size()`.
    ///
    /// Note that `new_layout.align()` need not be the same as `old_layout.align()`.
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
        &self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        debug_assert!(
            new_layout.size() <= old_layout.size(),
            "`new_layout.size()` must be smaller than or equal to `old_layout.size()`"
        );

        let new_ptr = self.allocate(new_layout)?;

        // SAFETY: because `new_layout.size()` must be lower than or equal to
        // `old_layout.size()`, both the old and new memory allocation are valid for reads and
        // writes for `new_layout.size()` bytes. Also, because the old allocation wasn't yet
        // deallocated, it cannot overlap `new_ptr`. Thus, the call to `copy_nonoverlapping` is
        // safe. The safety contract for `dealloc` must be upheld by the caller.
        unsafe {
            ptr::copy_nonoverlapping(ptr.as_ptr(), new_ptr.as_mut_ptr(), new_layout.size());
            self.deallocate(ptr, old_layout);
        }

        Ok(new_ptr)
    }

    /// Creates a "by reference" adapter for this instance of `Allocator`.
    ///
    /// The returned adapter also implements `Allocator` and will simply borrow this.
    #[inline(always)]
    fn by_ref(&self) -> &Self
    where
        Self: Sized,
    {
        self
    }
}

/// An [`Allocator`] that can be registered as the standard library’s default
/// through the `#[global_allocator]` attribute.
///
/// Types implementing this trait can be used as the default allocator for
/// memory allocations through `Box`, `Vec` and the collection types. For
/// instance, the `System` allocator implements this trait, and thus can be
/// explicitly set as the default like so:
/// ```
/// use std::alloc::System;
///
/// #[global_allocator]
/// static ALLOCATOR: System = System;
/// ```
///
/// The `Global` allocator forwards all memory allocation requests to the
/// `static` annotated with `#[global_allocator]`. Hence, `Global` does not
/// implement `GlobalAllocator` itself, as that would lead to infinite recursion.
///
/// # Note to implementors
///
/// This trait is used to prevent the infinite recursion that would occur if the
/// default allocator were to attempt to allocate memory through `Global` (and
/// thus from itself).
///
/// When to implement this trait:
/// * for custom global allocators that only use system memory allocation
///   services.
/// * for allocators that wrap another allocator that implements `GlobalAllocator`.
///
/// When **not** to implement this trait:
/// * for wrappers of arbitrary allocators (which might end up being `Global`,
///   leading to infinite recursion).
///
/// # Safety
///
/// In addition to the safety requirements of `Allocator`, global allocators are
/// subject to some additional constraints:
///
/// * It's undefined behavior if global allocators unwind. This restriction may
///   be lifted in the future, but currently a panic from any of these
///   functions may lead to memory unsafety.
///
/// * You must not rely on allocations actually happening, even if there are explicit
///   heap allocations in the source. The optimizer may detect unused allocations that it can either
///   eliminate entirely or move to the stack and thus never invoke the allocator. The
///   optimizer may further assume that allocation is infallible, so code that used to fail due
///   to allocator failures may now suddenly work because the optimizer worked around the
///   need for an allocation. More concretely, the following code example is unsound, irrespective
///   of whether your custom allocator allows counting how many allocations have happened.
///
///   ```rust,ignore (unsound and has placeholders)
///   drop(Box::new(42));
///   let number_of_heap_allocs = /* call private allocator API */;
///   unsafe { std::hint::assert_unchecked(number_of_heap_allocs > 0); }
///   ```
///
///   Note that the optimizations mentioned above are not the only
///   optimization that can be applied. You may generally not rely on heap allocations
///   happening if they can be removed without changing program behavior.
///   Whether allocations happen or not is not part of the program behavior, even if it
///   could be detected via an allocator that tracks allocations by printing or otherwise
///   having side effects.
///
/// # Re-entrance
///
/// When implementing a global allocator, one has to be careful not to create an infinitely recursive
/// implementation by accident, as many constructs in the Rust standard library may allocate in
/// their implementation. For example, on some platforms, [`std::sync::Mutex`] may allocate, so using
/// it is highly problematic in a global allocator.
///
/// For this reason, one should generally stick to library features available through
/// [`core`], and avoid using [`std`] in a global allocator. A few features from [`std`] are
/// guaranteed to not use `#[global_allocator]` to allocate:
///
///  - [`std::thread_local`],
///  - [`std::thread::current`],
///  - [`std::thread::park`] and [`std::thread::Thread`]'s [`unpark`] method and
/// [`Clone`] implementation.
///
/// [`std`]: ../../std/index.html
/// [`std::sync::Mutex`]: ../../std/sync/struct.Mutex.html
/// [`std::thread_local`]: ../../std/macro.thread_local.html
/// [`std::thread::current`]: ../../std/thread/fn.current.html
/// [`std::thread::park`]: ../../std/thread/fn.park.html
/// [`std::thread::Thread`]: ../../std/thread/struct.Thread.html
/// [`unpark`]: ../../std/thread/struct.Thread.html#method.unpark
#[unstable(feature = "allocator_api", issue = "32838")]
#[expect(multiple_supertrait_upcastable)]
pub unsafe trait GlobalAllocator: Allocator + Sync + 'static {}

#[unstable(feature = "allocator_api", issue = "32838")]
#[rustc_const_unstable(feature = "const_heap", issue = "79597")]
const unsafe impl<A> Allocator for &A
where
    A: [const] Allocator + ?Sized,
{
    #[inline]
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        (**self).allocate(layout)
    }

    #[inline]
    fn allocate_zeroed(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        (**self).allocate_zeroed(layout)
    }

    #[inline]
    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        // SAFETY: the safety contract must be upheld by the caller
        unsafe { (**self).deallocate(ptr, layout) }
    }

    #[inline]
    unsafe fn grow(
        &self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        // SAFETY: the safety contract must be upheld by the caller
        unsafe { (**self).grow(ptr, old_layout, new_layout) }
    }

    #[inline]
    unsafe fn grow_zeroed(
        &self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        // SAFETY: the safety contract must be upheld by the caller
        unsafe { (**self).grow_zeroed(ptr, old_layout, new_layout) }
    }

    #[inline]
    unsafe fn shrink(
        &self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        // SAFETY: the safety contract must be upheld by the caller
        unsafe { (**self).shrink(ptr, old_layout, new_layout) }
    }
}

#[unstable(feature = "allocator_api", issue = "32838")]
unsafe impl<A> Allocator for &mut A
where
    A: Allocator + ?Sized,
{
    #[inline]
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        (**self).allocate(layout)
    }

    #[inline]
    fn allocate_zeroed(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        (**self).allocate_zeroed(layout)
    }

    #[inline]
    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        // SAFETY: the safety contract must be upheld by the caller
        unsafe { (**self).deallocate(ptr, layout) }
    }

    #[inline]
    unsafe fn grow(
        &self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        // SAFETY: the safety contract must be upheld by the caller
        unsafe { (**self).grow(ptr, old_layout, new_layout) }
    }

    #[inline]
    unsafe fn grow_zeroed(
        &self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        // SAFETY: the safety contract must be upheld by the caller
        unsafe { (**self).grow_zeroed(ptr, old_layout, new_layout) }
    }

    #[inline]
    unsafe fn shrink(
        &self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        // SAFETY: the safety contract must be upheld by the caller
        unsafe { (**self).shrink(ptr, old_layout, new_layout) }
    }
}
