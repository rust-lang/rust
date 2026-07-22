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
/// ### Equivalent allocators
///
/// Multiple allocator values can sometimes be interchangeable with each other.
/// When this is the case, we refer to those allocators as being *equivalent* to
/// each other.
///
/// The following conditions are sufficient conditions for allocators to be equivalent.
/// * An allocator is equivalent to itself. (Equivalence is reflexive.)
/// * If an allocator is equivalent to a second allocator, then
///   the second allocator is also equivalent to the first. (Equivalence is symmetric.)
/// * If an allocator is equivalent to a second allocator, and
///   the second allocator is equivalent to a third allocator, then
///   the first allocator is also equivalent to the third allocator.
///   (Equivalence is transitive.)
/// * Moving, subtyping, unsize-coercing, or trait-upcasting an allocator does not change
///   what the allocator is equivalent to.
/// * Copying or cloning allocator results in an allocator that's
///   equivalent to the initial allocator, should the [`AllocatorClone`] trait
///   be implemented.
///
/// Additionally, implementors of `Allocator` may specify additional equivalences
/// between allocators. It is the responsibility of such implementors to make sure
/// that equivalent allocators have "compatible" `Allocator` implementations.
/// In particular, the standard library specifies the following equivalences:
/// * A reference to an allocator (either `&` or `&mut`) is equivalent to
///   the allocator being referenced.
/// * A `Box`, `Rc`, or `Arc` containing an allocator is equivalent to
///   the allocator inside.
/// * All `Global` allocator instances are equivalent with each other.
/// * All `System` allocator instances are equivalent with each other.
///
/// ### Currently allocated memory
///
/// Some of the methods require that a memory block is *currently allocated* by some specific allocator.
/// This means that:
/// * the starting address for that memory block was previously returned by
///   the [`allocate`], [`allocate_zeroed`], [`grow`], [`grow_zeroed`], or [`shrink`] methods,
///   called on an allocator that's equivalent to this specific allocator; and
/// * the memory block has not subsequently been [*invalidated*].
///
/// ### Invalidating memory blocks
///
/// A memory block that is currently allocated becomes *invalidated* when one
/// of the following happens:
/// * The memory block is deallocated. This occurs when the memory block
///   is passed as an argument to a [`deallocate`] call, or when it is passed
///   as an argument to a [`grow`], [`grow_zeroed`] or [`shrink`] call that returns `Ok`.
/// * All (equivalent) allocators that this memory block is allocated with,
///   each has one of the following happen to them:
///   * The allocator's destructor runs.
///   * The allocator is mutated through public API taking `&mut` access.
///   * One of the borrow-checker lifetimes in the allocator's type expires.
///
/// Note that these conditions imply that a collection may ensure that
/// any specific currently allocated memory block won't be invalidated, by:
/// * not deallocating that memory block,
/// * owning an allocator that memory block is allocated with, and
/// * not publicly exposing `&mut` access to that allocator.
///
/// Also note that safe public API of an allocator with `&` access is not
/// allowed to invalidate its memory blocks. Furthermore, unsafe public API
/// of an allocator with `&` access must document that they invalidate
/// memory blocks (e.g., by calling `deallocate`) if they do. Therefore,
/// collections may safely expose `&` access to its allocator.
///
/// Also note that, even in cases where are other "alive" allocators known to be
/// equivalent to a given collection's allocator, most collections still should
/// not publicly expose `&mut` access to its allocator. The fact that there are
/// other "alive" allocators would prevent this `&mut` access from invalidating
/// the collection's memory block, but public `&mut` access is still likely to
/// be unsound, since a user could replace the collection's allocator with
/// a non-equivalent allocator, causing the collection to deallocate its memory
/// with the wrong allocator.
///
/// [`allocate`]: Allocator::allocate
/// [`allocate_zeroed`]: Allocator::allocate_zeroed
/// [`grow`]: Allocator::grow
/// [`grow_zeroed`]: Allocator::grow_zeroed
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
///    - `max` is the actual size returned from [`allocate`], [`allocate_zeroed`],
///      [`grow`], [`grow_zeroed`], or [`shrink`].
///
/// [`layout.align()`]: Layout::align
/// [`layout.size()`]: Layout::size
///
/// # Safety
///
/// Implementors of `Allocator` must ensure that a memory block that
/// is [*currently allocated*] by the allocator points to valid memory,
/// until that memory block is [*invalidated*]. The implementor must also
/// not violate this invariant of `Allocator` via allocator equivalences
/// that are in the implementor's control (e.g., via an incorrect `unsafe
/// impl AllocatorClone for MyAllocator`).
///
/// More concretely, the following code example is unsound, irrespective of whether your custom
/// allocator allows counting how many allocations have happened:
///
/// ```rust,ignore (unsound and has placeholders)
/// drop(Box::new_in(42, MyCustomAllocator));
/// let number_of_heap_allocs = /* call private allocator API */;
/// unsafe { std::hint::assert_unchecked(number_of_heap_allocs > 0); }
/// ```
///
/// Additionally, any memory block returned by the allocator must
/// satisfy the allocation invariants described in `core::ptr`.
/// In particular, if a block has base address `p` and size `n`,
/// then `p as usize + n <= usize::MAX` must hold.
///
/// This ensures that pointer arithmetic within the allocation
/// (for example, `ptr.add(len)`) cannot overflow the address space.
///
/// Lastly, none of the allocating or deallocating methods may unwind. This restriction
/// may be lifted in the future by ensuring unwinding out of an allocating function always
/// aborts.
///
/// [*currently allocated*]: #currently-allocated-memory
/// [*invalidated*]: #invalidating-memory-blocks
// NOTE: the above bound on allocating methods not unwinding, alongside the similar
// bound on `AllocatorClone`, are currently load-bearing in std! see the below issues
// and make sure they cannot be triggered before relaxing this:
// https://rust.tf/156490
// https://rust.tf/155746
#[unstable(feature = "allocator_api", issue = "32838")]
#[rustc_const_unstable(feature = "const_heap", issue = "79597")]
pub const unsafe trait Allocator {
    /// Attempts to allocate a block of memory.
    ///
    /// On success, returns a [`NonNull<[u8]>`][NonNull] meeting the size and alignment
    /// guarantees of `layout`. The returned block may have a larger size than specified
    /// by `layout.size()`, and may or may not have its contents initialized.
    ///
    /// It is recommended that overallocating as per the above is only performed if doing so
    /// is cheap; there is no guarantee that the caller is able to take advantage of the
    /// returned excess. Implementors are free to e.g. provide an alternate method to query
    /// available excess if doing so is expensive and should be left to the caller.
    ///
    /// Note that the returned block of memory is considered [*currently allocated*]
    /// with this allocator (and equivalent allocators).
    /// Therefore, it is the responsibility of implementors of `Allocator` to make sure that
    /// this block of memory points to valid memory until the block is [*invalidated*]
    ///
    /// [*currently allocated*]: #currently-allocated-memory
    /// [*invalidated*]: #invalidating-memory-blocks
    ///
    /// # Errors
    ///
    /// Returning `Err` indicates that either memory is exhausted or `layout` does not meet
    /// allocator's size or alignment constraints.
    ///
    /// Implementations are encouraged to return `Err` on memory exhaustion rather than
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
    /// Implementations are encouraged to return `Err` on memory exhaustion rather than
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
    /// Note that it is *immediate* language UB for a deallocation or reallocation to
    /// invalidate any outstanding references, smart pointers, etc.; thus, notably, an
    /// allocator that has been moved into its own [*currently allocated*] memory may
    /// not have its backing memory be freed, even if the allocator is never used again
    /// afterwards. This is due to the fact that such a deallocation would invalidate the
    /// `&self` reference passed to this method.
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
    /// If this returns `Ok`, then the memory block referenced by `ptr` has been [*invalidated*].
    /// The old `ptr` must not be used to access the memory, even if the allocation was grown in-place.
    /// The newly returned pointer is the only valid pointer for accessing this memory now.
    /// All bytes past `old_layout.size()` should be assumed to be uninitialised.
    ///
    /// If this method returns `Err`, then the memory block has not been *invalidated*,
    /// and the contents of the memory block are unaltered.
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
    /// [*invalidated*]: #invalidating-memory-blocks
    ///
    /// # Errors
    ///
    /// Returns `Err` if the new layout does not meet the allocator's size and alignment
    /// constraints of the allocator, or if growing otherwise fails.
    ///
    /// Implementations are encouraged to return `Err` on memory exhaustion rather than
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
    ///   * Bytes `old_layout.size()..new_size` are zeroed. `new_size` refers to the size
    ///     of the memory block returned by the `grow_zeroed` call, which may be larger than
    ///     `new_layout.size()`.
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
    /// Implementations are encouraged to return `Err` on memory exhaustion rather than
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
    ///
    /// If this returns `Ok`, then the memory block referenced by `ptr` has been [*invalidated*].
    /// The old `ptr` must not be used to access the memory, even if the allocation was shrunk in-place.
    /// The newly returned pointer is the only valid pointer for accessing this memory now.
    /// All bytes past `new_layout.size()` should be assumed to be uninitialised.
    ///
    /// If this method returns `Err`, then the memory block has not been *invalidated*,
    /// and the contents of the memory block are unaltered.
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
    /// [*invalidated*]: #invalidating-memory-blocks
    ///
    /// # Errors
    ///
    /// Returns `Err` if the new layout does not meet the allocator's size and alignment
    /// constraints of the allocator, or if shrinking otherwise fails.
    ///
    /// Implementations are encouraged to return `Err` on memory exhaustion rather than
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
/// When implementing a global allocator, one has to be careful not to create an infinitely
/// recursive implementation by accident, as many constructs in the Rust standard library may
/// allocate in their implementation. For example, on some platforms, [`std::sync::Mutex`] may
/// allocate, so using it is highly problematic in a global allocator.
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
pub unsafe trait GlobalAllocator: StaticAllocator + Sync + 'static {}

/// Marks a type's [`Clone`] implementation as sound with regard to [`Allocator`] equivalence.
/// Implementors must ensure that, upon cloning, the two allocators are equivalent
/// (i.e. it is possible to free memory with one that was allocated with the other).
/// Further, mutable accesses such as moving or dropping the allocator must not invalidate
/// its currently allocated blocks at least so long as clones exist.
///
/// Additionally, the bound that allocators do not unwind when (de)allocating also applies
/// to guaranteeing allocators will not unwind when cloned. This bound trivially holds for
/// allocators that are `Copy`, assuming the `Clone` implementation is not different.
#[unstable(feature = "allocator_api", issue = "32838")]
pub unsafe trait AllocatorClone: Allocator + Clone {}

/// Marks a type's [`PartialEq`] implementation as sound with regard to [`Allocator`] equivalence.
/// Implementors must ensure that, upon equality, the two allocators are equivalent
/// (i.e. it is possible to free memory with one that was allocated with the other), and
/// that the two allocators behave "as if" they are clones of each other as per
/// [`AllocatorClone`].
#[unstable(feature = "allocator_api", issue = "32838")]
pub unsafe trait AllocatorEq<T = Self>: Allocator + PartialEq<T>
where
    T: ?Sized + AllocatorEq<Self>,
{
}

/// Marks that an allocator and its supertypes will never invalidate currently allocated
/// memory unless explicitly deallocated via a call to a deallocating method, even if
/// dropped or if the allocator's lifetime expires.
///
/// This is a necessity in conjunction with [`Pin`], as only allocators that promise
/// memory is never reused without a destructor running may be used to back a pinned pointer.
///
/// # Safety
///
/// Implementors must ensure that memory cannot be freed except via a call to
/// `Allocator::deallocate`, and that subtype coercion preserves this invariant.
///
/// These requirements trivially apply to allocators that always maintain global state, such as
/// `System` or `Global`. However, due to subtype coercion, it is *not* sound to implement
/// for an arbitrary `Allocator + 'static` due to [edge-case interactions][unsound] with
/// `Pin::clone`. Namely, an impl of `StaticAllocator for MyAllocator + 'long` guarantees that an
/// impl of `StaticAllocator for MyAllocator + 'short` would be sound to write.
///
/// The following must thus be guaranteed:
/// - the `Drop` impl of the allocator does not invalidate any allocations;
/// - the allocator does not expose a safe API surface that allows invalidating
///   its allocations;
/// - the allocator's lifetime expiring does not invalidate any allocations;
/// - the above also hold for all equivalent allocators (see [`Allocator`] docs).
///
/// [`Pin`]: ../../core/pin/struct.Pin.html
/// [unsound]: https://github.com/rust-lang/rust/issues/157089
#[unstable(feature = "allocator_api", issue = "32838")]
pub unsafe trait StaticAllocator: Allocator {}

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

#[unstable(feature = "allocator_api", issue = "32838")]
unsafe impl<A: Allocator + ?Sized> AllocatorClone for &A {}

// FIXME: is this impl sound? It would be insta-stable behaviour once `Box::pin_in` is stable.
// see https://rust.tf/157089
//#[unstable(feature = "allocator_api", issue = "32838")]
//unsafe impl<A: StaticAllocator + ?Sized> StaticAllocator for &A {}
