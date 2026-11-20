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
#[allow(deprecated)]
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
#[stable(feature = "allocator_api", since = "1.100.0")]
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub struct AllocError;

#[stable(feature = "allocator_api", since = "1.100.0")]
impl Error for AllocError {}

// (we need this for downstream impl of trait Error)
#[stable(feature = "allocator_api", since = "1.100.0")]
impl fmt::Display for AllocError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("memory allocation failed")
    }
}

/// An implementation of `Allocator` can allocate, grow, shrink, and deallocate arbitrary blocks of
/// data described via [`Layout`][].
///
/// `Allocator` is mostly designed to be implemented on ZSTs, references, or smart pointers,
/// but can also be implemented directly on the underlying memory-owning type so long as it
/// upholds the necessary guarantees. In general, an allocator of the type `MyAlloc([u8; N])`
/// cannot be soundly created without being pinned or otherwise immovable in order to be
/// correct.
///
/// In contrast to [`GlobalAlloc`][], `Allocator` allows zero-sized allocations. If an underlying
/// allocator does not support this (like jemalloc) or responds by returning a null pointer
/// (such as `libc::malloc`), this must be caught by the implementation.
///
/// # Examples
///
/// Sample that keeps track of the number of existing allocations:
///
/// ```
/// use std::alloc::{AllocError, Allocator, Layout, System};
/// use std::ptr::NonNull;
/// use std::sync::atomic::{AtomicUsize, Ordering};
///
/// static LIVE_ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);
///
/// struct CountingAllocator;
///
/// unsafe impl Allocator for CountingAllocator {
///     fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
///         let ptr = System.allocate(layout)?;
///         LIVE_ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
///         Ok(ptr)
///     }
///
///     unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
///         // SAFETY: the caller guarantees `ptr` was allocated by this allocator with `layout`,
///         // and every block from this allocator was allocated by `System` with the same layout.
///         unsafe { System.deallocate(ptr, layout) };
///         LIVE_ALLOCATIONS.fetch_sub(1, Ordering::Relaxed);
///     }
/// }
///
/// // The Box allocates through our allocator.
/// let new_box = Box::new_in(10, CountingAllocator);
/// assert_eq!(LIVE_ALLOCATIONS.load(Ordering::Relaxed), 1);
/// drop(new_box);
/// assert_eq!(LIVE_ALLOCATIONS.load(Ordering::Relaxed), 0);
/// ```
///
/// ## Implementation Requirements
///
/// In order to be usable in a flexible manner while still being sound, implementors of the trait
/// must uphold very detailed semantics as explained below; the following terms are thus provided
/// as vocabulary for allocator safety and implementation requirements:
///
/// ### Equivalent allocators
///
/// Multiple allocator values can sometimes be interchangeable with each other.
/// When this is the case, we refer to those allocators as being *equivalent* to
/// each other.
///
/// Users of allocators may assume the following are true of equivalent allocators,
/// and implementors must ensure these rules are upheld:
/// * An allocator is equivalent to itself. (Equivalence is reflexive.)
/// * If an allocator is equivalent to a second allocator, then
///   the second allocator is also equivalent to the first. (Equivalence is symmetric.)
/// * If an allocator is equivalent to a second allocator, and
///   the second allocator is equivalent to a third allocator, then
///   the first allocator is also equivalent to the third allocator.
///   (Equivalence is transitive.)
/// * Moving, subtyping, unsize-coercing, or trait-upcasting an allocator does not change
///   what the allocator is equivalent to.
/// * Copying or cloning an allocator creates an equivalent one, should the
///   [`AllocatorClone`] trait be implemented.
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
/// ### Memory Ownership
///
/// For an allocator to return a memory block, it is required to have *ownership* of it.
/// The act of returning a pointer to that block from one of the allocating methods
/// causes ownership of it to be transferred to the caller.
///
/// Ownership is exclusive. It is undefined behavior for the allocator to read, write, invalidate,
/// or return any memory block that it has given up ownership of, until ownership is regained.
/// This memory block is owned by the caller; the allocator must not touch or reuse it.
///
/// An allocator regains ownership of a memory block when one of the following happens:
/// * The memory block is deallocated. This occurs when the memory block
///   is passed as an argument to a [`deallocate`] call, or when it is passed
///   as an argument to a [`grow`], [`grow_zeroed`] or [`shrink`] call that returns `Ok`.
/// * For all equivalent instances of this allocator, at least one of the following has occurred:
///   * The allocator's destructor runs.
///   * The allocator is mutated through a public or otherwise untrusted API taking `&mut` access.
///   * One of the borrow-checker lifetimes in the allocator's type expires.
///
/// Ownership can only be passed back to the allocator that the memory block was acquired from (or
/// equivalent allocators). If the allocator got the memory from some inner allocator, you cannot
/// "skip" the outer allocator by deallocating directly with the inner one.
///
/// When an allocator regains ownership of a memory block, the consumer cannot use it anymore; only
/// the allocator can touch it.
///
/// Note that some allocators may choose not to reclaim memory blocks in certain situations.
/// In such cases, the allocator may document that ownership remains with the consumer.
/// For example, `Arc<Global>` is an allocator with a destructor, but that destructor
/// does nothing relevant to the `Allocator` API, because `Global` has no destructor;
/// therefore, no memory will be reclaimed when the allocator itself goes out of scope.
///
/// #### Collections
///
/// The conditions for reclaiming [*ownership*] imply that a collection may use a memory block for
/// storage by:
/// * not deallocating that memory block,
/// * owning the allocator that the memory block was acquired from (or an equivalent allocator), and
/// * not publicly exposing `&mut` access to that allocator.
///
/// Note that simply holding a pointer to the memory block is generally not enough, because the
/// collection may lose ownership when the allocator is mutated or dropped independently.
///
/// Even in cases where there are other known-alive equivalent allocators, collections still should
/// not publicly expose `&mut` access to their allocator, because such API can be used to replace
/// the allocator, causing undefined behavior when memory blocks are deallocated with the wrong
/// allocator.
///
/// Safe public API of an allocator with `&` access cannot be used to deallocate memory, or pass
/// ownership of a memory block to the allocator. Furthermore, unsafe public API of an allocator
/// with `&` access must document their effects on ownership and validity of affected memory
/// (for example, allocations from [`Global`] can be deallocated using the free-standing [`dealloc`]
/// function).
///
/// Therefore, a collection may safely expose `&` access to its allocator to untrusted code.
///
/// Note that if you move an allocator into a memory block acquired from itself, you cannot
/// deallocate that memory block or drop the allocator (if its destructor would reclaim the block),
/// until you've moved it back out of the block.
/// Doing so would cause the `&(mut) self` parameter of the allocator API to be derived from the
/// pointer to the memory block; this is not considered exclusive ownership over that block,
/// and could cause immediate undefined behavior if the allocator invalidates memory that a
/// `&(mut) self` parameter is still pointing to.
/// It is generally still okay to deallocate other memory blocks, or use an equivalent allocator
/// (that doesn't defer to the one in the block) to deallocate it.
///
/// #### Wrapping Allocators
///
/// It may make sense for an allocator to get its memory from a different allocator.
/// Such allocators can often be written as a wrapper around a generic allocator type.
///
/// Any such allocator that wants to hand out memory from another allocator should follow
/// the same [steps](#collections) for ensuring validity as collections do.
/// Otherwise, it may lack control over when the last instance of the inner allocator is
/// mutated, and can thus not guarantee adequate [*ownership*] to its callers.
///
/// This naturally results in a chain of ownership, where allocators can wrap other allocators, passing
/// on ownership of parts of the memory acquired from inner allocators as needed.
///
/// Note that in a chain of wrapping allocators, it may be possible to create cycles. For example, some
/// allocator `A` may end up wrapping allocator `B`, which wraps `&A` again. This can cause `A` to hand
/// out memory from itself, which may appear to `B` as if `A` was handing out pointers to the same
/// memory twice. This is allowed, as `A` has also transferred ownership of the memory to itself.
///
/// For this reason, it is generally incorrect to assume that memory blocks from an allocator
/// do not overlap, unless you currently own both blocks.
///
/// ### Memory fitting
///
/// Some of the methods require that a `layout` *fits* a memory block or vice versa. This means
/// that the following conditions must hold:
///  * the memory block was acquired from one of the allocating methods,
///  * [`layout.align()`] must be the same as the alignment of the layout used to allocate the block, and
///  * [`layout.size()`] must fall in the range `min ..= max`, where:
///    - `min` is the size of the layout used to allocate the block, and
///    - `max` is the actual size returned from [`allocate`], [`allocate_zeroed`],
///      [`grow`], [`grow_zeroed`], or [`shrink`].
///
/// # Safety
///
/// Implementors of `Allocator` must ensure that every memory block that
/// they return from one of the allocating methods points to valid memory
/// until they reclaim [*ownership*] over that memory block. Note that resizing
/// methods like [`grow`] only temporarily take ownership of the memory block
/// if resizing fails; it must remain valid when this happens.
///
/// The implementor must also not violate any invariants of `Allocator` via allocator
/// equivalences that are in the implementor's control.
///
/// Additionally, any memory block returned by the allocator must
/// satisfy the allocation invariants described in `core::ptr`.
/// In particular, if a block has base address `p` and size `n`,
/// then `p as usize + n <= usize::MAX` must hold.
/// This ensures that pointer arithmetic within the block (for example, `ptr.add(len)`) cannot
/// overflow the address space.
///
/// None of the allocating or deallocating methods may unwind. This restriction
/// may be lifted in the future by ensuring unwinding out of an allocating function always
/// aborts. If an implementor of `Allocator` also has drop glue or directly implements `Drop`,
/// dropping the allocator must not result in an unwind.
///
/// The methods on this trait must be *correct*; in particular, ownership and validity
/// must be handled as documented, the layout requested must be respected, calls must zero
/// out memory if the documentation so requires, returning an `AllocError` from a reallocating
/// method must ensure that ownership is passed back to the caller, and de/reallocating
/// calls must accept layouts in the ranges defined by their documentation.
///
/// Implementors must also uphold the overall `Allocator` contract described in the previous
/// sections.
///
/// [`allocate`]: Allocator::allocate
/// [`allocate_zeroed`]: Allocator::allocate_zeroed
/// [`dealloc`]: ../../alloc/alloc/fn.dealloc.html
/// [`deallocate`]: Allocator::deallocate
/// [`grow`]: Allocator::grow
/// [`grow_zeroed`]: Allocator::grow_zeroed
/// [`Global`]: ../../alloc/alloc/struct.Global.html
/// [`layout.align()`]: Layout::align
/// [`layout.size()`]: Layout::size
/// [*ownership*]: #memory-ownership
/// [`shrink`]: Allocator::shrink
#[stable(feature = "allocator_api", since = "1.100.0")]
#[rustc_const_unstable(feature = "const_heap", issue = "79597")]
pub const unsafe trait Allocator {
    /// Attempts to allocate a block of memory.
    ///
    /// On success, returns a [`NonNull<[u8]>`][NonNull] meeting the size and alignment
    /// guarantees of `layout`. The returned block may have a larger size than specified
    /// by `layout.size()`, and may or may not have its contents initialized.
    /// Use [`allocate_zeroed`] if zero-initialization is needed.
    ///
    /// It is recommended that overallocating as per the above is only performed if doing so
    /// is cheap; there is no guarantee that the caller is able to take advantage of the
    /// returned excess. Implementors are free to e.g. provide an alternate method to query
    /// available excess if doing so is expensive and should be left to the caller.
    ///
    /// The caller gains [*ownership*] of the returned memory block, and the allocator
    /// guarantees that this is respected.
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
    /// # Examples
    ///
    /// ```
    /// use std::alloc::{Allocator, Layout, System};
    ///
    /// let s = "Hello, world!";
    /// let layout = Layout::array::<u8>(s.len()).unwrap();
    // FIXME(slice_ptr_get): use `.as_non_null_ptr()` once stable
    /// let ptr = System.allocate(layout).expect("allocation failed").cast::<u8>();
    ///
    /// // SAFETY: the block is at least `s.len()` bytes long and was just allocated,
    /// // so it cannot overlap `s`.
    /// unsafe { ptr.as_ptr().copy_from_nonoverlapping(s.as_ptr(), s.len()) };
    ///
    /// // SAFETY: the first `s.len()` bytes were initialized by the copy above.
    /// let bytes = unsafe { std::slice::from_raw_parts(ptr.as_ptr(), s.len()) };
    /// assert_eq!(bytes, s.as_bytes());
    ///
    /// // SAFETY: `ptr` was allocated by `System` with `layout`.
    /// unsafe { System.deallocate(ptr, layout) };
    /// ```
    ///
    /// [`handle_alloc_error`]: ../../alloc/alloc/fn.handle_alloc_error.html
    /// [*ownership*]: #memory-ownership
    /// [`allocate_zeroed`]: Allocator::allocate_zeroed
    #[stable(feature = "allocator_api", since = "1.100.0")]
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError>;

    /// Attempts to allocate a zero-initialized block of memory.
    ///
    /// This method has the same contract as [`allocate`], with the addition that the returned
    /// memory is guaranteed to be zero-initialized, for the full returned length.
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
    /// # Examples
    ///
    /// ```
    /// use std::alloc::{Allocator, Layout, System};
    ///
    /// let layout = Layout::new::<u64>();
    /// let ptr = System.allocate_zeroed(layout).expect("allocation failed");
    ///
    /// // N.B. the below would be unsound with `allocate` rather than `allocate_zeroed`.
    /// // SAFETY: `allocate_zeroed` returns a valid, initialized block.
    /// let bytes = unsafe { ptr.as_ref() };
    /// assert!(bytes.iter().all(|&b| b == 0));
    ///
    /// // SAFETY: `ptr` was allocated by `System` with `layout`.
    // FIXME(slice_ptr_get): use `.as_non_null_ptr()` once stable
    /// unsafe { System.deallocate(ptr.cast(), layout) };
    /// ```
    ///
    /// [`handle_alloc_error`]: ../../alloc/alloc/fn.handle_alloc_error.html
    /// [`allocate`]: Allocator::allocate
    #[stable(feature = "allocator_api", since = "1.100.0")]
    fn allocate_zeroed(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        let ptr = self.allocate(layout)?;
        // SAFETY: `alloc` returns a valid memory block
        unsafe { ptr.as_non_null_ptr().as_ptr().write_bytes(0, ptr.len()) }
        Ok(ptr)
    }

    /// Deallocates the memory block referenced by `ptr`.
    ///
    /// This transfers [*ownership*] of the block back to the allocator.
    ///
    /// # Safety
    ///
    /// * `ptr` must point to a block of memory acquired from this allocator (or an equivalent one),
    /// * `layout` must [*fit*] that block of memory, and
    /// * the ownership contract of the allocator must be upheld.
    ///
    /// [*fit*]: #memory-fitting
    /// [*ownership*]: #memory-ownership
    #[stable(feature = "allocator_api", since = "1.100.0")]
    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout);

    /// Attempts to extend a memory block.
    ///
    /// When this method is called, [*ownership*] of the memory block is transferred back to the allocator.
    ///
    /// Returns a new [`NonNull<[u8]>`][NonNull] containing a pointer and the actual size of the allocated
    /// memory. The pointer is suitable for holding data described by `new_layout`. To accomplish
    /// this, the allocator may extend the memory block referenced by `ptr` to fit the new layout.
    ///
    /// If this returns `Ok`, then ownership of the memory block referenced by `ptr` remains with the allocator
    /// (except for overlapping parts of the returned memory block, if any).
    /// The old `ptr` must not be used to access the memory, even if the memory block was grown in-place.
    /// The newly returned pointer is the only valid pointer for accessing this memory now; ownership
    /// is transferred through it, with the updated layout.
    ///
    /// The first `old_layout.size()` bytes of the old memory block are moved to the new block by
    /// the allocator. The contents of the memory beyond that are unspecified, and should be
    /// assumed uninitialized; use [`grow_zeroed`] to zero-initialize the new bytes.
    ///
    /// If this method returns `Err`, then the caller regains ownership of the (entire) memory block.
    /// Its contents are unaltered, and `ptr` is still valid for reads and writes on it.
    ///
    /// # Safety
    ///
    /// * `ptr` must point to a block of memory acquired from this allocator (or an equivalent one),
    /// * `old_layout` must [*fit*] that block of memory (the `new_layout` argument need not fit it),
    /// * `new_layout.size()` must be greater than or equal to `old_layout.size()`, and
    /// * the ownership contract of the allocator must be upheld.
    ///
    /// Note that `new_layout.align()` may be different from `old_layout.align()`.
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
    ///
    /// # Examples
    ///
    /// ```
    /// use std::alloc::{Allocator, Layout, System};
    ///
    /// let s1 = "Hello";
    /// let s2 = ", world!";
    ///
    /// let old_layout = Layout::array::<u8>(s1.len()).unwrap();
    // FIXME(slice_ptr_get): use `.as_non_null_ptr()` once stable
    /// let ptr = System.allocate(old_layout).expect("allocation failed").cast::<u8>();
    ///
    /// // SAFETY: the block is at least `s1.len()` bytes long and was just allocated,
    /// // so it cannot overlap `s1`.
    /// unsafe { ptr.as_ptr().copy_from_nonoverlapping(s1.as_ptr(), s1.len()) };
    ///
    /// let new_layout = Layout::array::<u8>(s1.len() + s2.len()).unwrap();
    ///
    /// // N.B. the original `ptr` is invalidated by this operation
    /// // SAFETY: `ptr` was allocated in `System` with `old_layout`,
    /// // and `new_layout` is at least as large as `old_layout`.
    // FIXME(slice_ptr_get): use `.as_non_null_ptr()` once stable
    /// let ptr = unsafe { System.grow(ptr, old_layout, new_layout) }
    ///     .expect("grow failed").cast::<u8>();
    ///
    /// // SAFETY: the block is at least `s1.len() + s2.len()` bytes long, so writing
    /// // `s2.len()` bytes at offset `s1.len()` stays in bounds; it cannot overlap `s2`.
    /// unsafe {
    ///     ptr.as_ptr().add(s1.len()).copy_from_nonoverlapping(s2.as_ptr(), s2.len());
    /// }
    ///
    /// // SAFETY: all bytes within `new_layout.size()` are now initialized.
    /// let bytes = unsafe { std::slice::from_raw_parts(ptr.as_ptr(), new_layout.size()) };
    /// assert_eq!(bytes, b"Hello, world!");
    ///
    /// // SAFETY: `ptr` was allocated in `System` with `new_layout`.
    /// unsafe { System.deallocate(ptr, new_layout) };
    /// ```
    ///
    /// [`handle_alloc_error`]: ../../alloc/alloc/fn.handle_alloc_error.html
    /// [*fit*]: #memory-fitting
    /// [*ownership*]: #memory-ownership
    /// [`grow_zeroed`]: Allocator::grow_zeroed
    #[stable(feature = "allocator_api", since = "1.100.0")]
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
        // `old_layout.size()`, both the old and new memory block are valid for reads and
        // writes for `old_layout.size()` bytes. Because the method holds exclusive ownership
        // of both memory blocks, they cannot overlap. Thus, the call to `copy_nonoverlapping` is
        // safe. The safety contract for `dealloc` must be upheld by the caller.
        unsafe {
            ptr::copy_nonoverlapping(ptr.as_ptr(), new_ptr.as_mut_ptr(), old_layout.size());
            self.deallocate(ptr, old_layout);
        }

        Ok(new_ptr)
    }

    /// Behaves like [`grow`], but also ensures that the new contents are set to zero before being
    /// returned.
    ///
    /// This method has the same contract as [`grow`], with the following additions:
    ///
    /// The memory block will contain the following contents after a successful call to
    /// `grow_zeroed`:
    ///   * Bytes `0..old_layout.size()` are moved from the original memory block.
    ///   * Bytes `old_layout.size()..new_size` are zeroed. `new_size` refers to the size
    ///     of the memory block returned by the `grow_zeroed` call, which may be larger than
    ///     `new_layout.size()`.
    ///
    /// # Safety
    ///
    /// Same as [`grow`].
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
    /// [*fit*]: #memory-fitting
    /// [`grow`]: Allocator::grow
    #[stable(feature = "allocator_api", since = "1.100.0")]
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
        // `old_layout.size()`, both the old and new memory block are valid for reads and
        // writes for `old_layout.size()` bytes. Because the method holds exclusive ownership
        // of both memory blocks, they cannot overlap. Thus, the call to `copy_nonoverlapping` is
        // safe. The safety contract for `dealloc` must be upheld by the caller.
        unsafe {
            ptr::copy_nonoverlapping(ptr.as_ptr(), new_ptr.as_mut_ptr(), old_layout.size());
            self.deallocate(ptr, old_layout);
        }

        Ok(new_ptr)
    }

    /// Attempts to shrink a memory block.
    ///
    /// Returns a new [`NonNull<[u8]>`][NonNull] containing a pointer and the actual size of the allocated
    /// memory. The pointer is suitable for holding data described by `new_layout`. To accomplish
    /// this, the allocator may shrink the memory block referenced by `ptr` to fit the new layout.
    ///
    /// When this method is called, [*ownership*] of the memory is transferred back to the allocator.
    ///
    /// If this returns `Ok`, then ownership of the memory block referenced by `ptr` remains with the allocator
    /// (except for overlapping parts of the returned memory block, if any).
    /// The old `ptr` must not be used to access the memory, even if the memory block was resized in-place.
    /// The newly returned pointer is the only valid pointer for accessing this memory now; ownership
    /// is transferred through it, with the updated layout.
    ///
    /// Note that the returned size of this method can be arbitrarily large, so it may not actually
    /// shrink the memory block. The main effect of this method is that future calls to the allocator
    /// get to use the smaller layout, even if the allocator returns a block that isn't smaller.
    /// The first `new_layout.size()` (which may be less than the returned size) bytes of the old memory
    /// block are moved to the start of the new block by the allocator.
    ///
    /// Even if the memory block was shrunk in-place, anything past the returned size may not be accessed. The
    /// caller does not own that memory; the new pointer might not even be valid for accesses beyond the new size.
    ///
    /// If this method returns `Err`, then the caller regains ownership of the (entire) memory block.
    /// Its contents are unaltered, and `ptr` is still valid for reads and writes on it.
    ///
    /// # Safety
    ///
    /// * `ptr` must point to a block of memory acquired from this allocator (or an equivalent one),
    /// * `old_layout` must [*fit*] that block of memory (the `new_layout` argument need not fit it),
    /// * `new_layout.size()` must be less than or equal to `old_layout.size()`, and
    /// * the ownership contract of the allocator must be upheld.
    ///
    /// Note that `new_layout.align()` may be different from `old_layout.align()`.
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
    /// # Examples
    ///
    /// ```
    /// use std::alloc::{Allocator, Layout, System};
    ///
    /// let old_layout = Layout::array::<u32>(4).unwrap();
    /// let ptr = System.allocate(old_layout).expect("allocation failed");
    ///
    /// // SAFETY: the block holds at least 4 `u32`s and is aligned for `u32`.
    /// unsafe { ptr.cast::<[u32; 4]>().write([7, 8, 9, 10]) };
    ///
    /// let new_layout = Layout::array::<u32>(2).unwrap();
    ///
    /// // N.B. the original `ptr` is invalidated by this operation
    /// // SAFETY: `ptr` was allocated in `System` with `old_layout`,
    /// // and `new_layout` is no larger than `old_layout`.
    /// let ptr = unsafe { System.shrink(ptr.cast(), old_layout, new_layout) }
    ///     .expect("shrink failed");
    ///
    /// // The contents that fit in the new layout are preserved.
    /// // SAFETY: the first two `u32`s were written above and `shrink` kept them.
    /// assert_eq!(unsafe { ptr.cast::<[u32; 2]>().read() }, [7, 8]);
    ///
    /// // SAFETY: `ptr` was allocated in `System` with `new_layout`.
    /// unsafe { System.deallocate(ptr.cast(), new_layout) };
    /// ```
    ///
    /// [*fit*]: #memory-fitting
    /// [*ownership*]: #memory-ownership
    /// [`handle_alloc_error`]: ../../alloc/alloc/fn.handle_alloc_error.html
    #[stable(feature = "allocator_api", since = "1.100.0")]
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
        // `old_layout.size()`, both the old and new memory block are valid for reads and
        // writes for `new_layout.size()` bytes. Because the method holds exclusive ownership
        // of both memory blocks, they cannot overlap. Thus, the call to `copy_nonoverlapping` is
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
#[unstable(feature = "allocator_ext", issue = "163177", implied_by = "allocator_api")]
#[expect(multiple_supertrait_upcastable)]
pub unsafe trait GlobalAllocator: StaticAllocator + Sync + 'static {}

/// Marks a type's [`Clone`] implementation as sound with regard to [`Allocator`] equivalence.
/// Implementors must ensure that, upon cloning, the two allocators are equivalent
/// (i.e. it is possible to free memory with one that was allocated with the other).
///
/// This means that as long as clones exist, mutating or dropping only one allocator does not
/// cause ownership of its allocated blocks to be reclaimed.
///
/// Additionally, the bound that allocators do not unwind when (de)allocating also applies
/// to guaranteeing allocators will not unwind when cloned.
///
/// It must also be the case that types which are `AllocatorClone` are either explicitly not
/// copyable (such as by containing a `!Copy` field) or that copying them also respects allocator
/// equivalence as if it had been a clone.
#[unstable(feature = "allocator_ext", issue = "163177", implied_by = "allocator_api")]
pub unsafe trait AllocatorClone: Allocator + Clone {}

/// Marks that an allocator and its supertypes will never reclaim [*ownership*] of memory that is
/// not explicitly passed to one of its de-/reallocating methods.
///
/// Even if the allocator is dropped or one of its lifetimes expires, it is guaranteed that ownership
/// of the memory acquired from this allocator remains with the current owner.
///
/// Equivalent allocators may still be used to reclaim memory.
///
/// This is a necessity in conjunction with [`Pin`], as only allocators that promise
/// memory is never reused without a destructor running may be used to back a pinned pointer.
///
/// # Safety
///
/// Implementors must not reclaim ownership of memory blocks acquired from the allocator, unless
/// they are passed to one of the de-/reallocating methods of [`Allocator`].
///
/// This must be upheld by all instances of the type this is implemented on, as well as by *all*
/// instances of all supertypes of this type (not just instances acquired from subtype coercion).
///
/// For example, if `A` is an allocator that reclaims memory on drop, and memory blocks from `&'static A`
/// are only reclaimed by explicit deallocation calls, then `&'static A` still *cannot* implement
/// this trait. This is because an instance of `A` can be used to allocate with `&A`, and then
/// drop `A` to reclaim the memory without explicit deallocation. `&A` is a supertype of `&'static A`, so
/// this violates the "all instances of all supertypes" requirement.
///
/// This requirement is very strict, to guard against [edge-case interactions][pin-clone-issue] with e.g. `Pin::clone`.
///
/// In particular, the following must thus be guaranteed:
/// - the destructor of the allocator does not reclaim allocated memory;
/// - the allocator does not expose safe API (even through `&mut`) that allows reclaiming allocated memory;
/// - the allocator's lifetime expiring does not allow an equivalent allocator (e.g. `A` for `&'a A`)
///   to reclaim allocated memory; and
/// - the above also hold for all equivalent allocators (see [`Allocator`] docs).
///
/// The safety requirements of this trait trivially apply to allocators like `System` or `Global`.
///
/// [`Pin`]: core::pin::Pin
/// [*ownership*]: Allocator#memory-ownership
/// [pin-clone-issue]: https://github.com/rust-lang/rust/issues/157089
#[unstable(feature = "allocator_ext", issue = "163177", implied_by = "allocator_api")]
pub unsafe trait StaticAllocator: Allocator {}

#[stable(feature = "allocator_api", since = "1.100.0")]
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

#[stable(feature = "allocator_api", since = "1.100.0")]
#[rustc_const_unstable(feature = "const_heap", issue = "79597")]
const unsafe impl<A> Allocator for &mut A
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

/// Defers to the pinned allocator. Requires that the smart pointer inside the
/// [`Pin`](core::pin::Pin) is well-behaved.
///
/// This trait impl is unstable and only usable on nightly with `feature(allocator_ext)`.
/// It may be changed or removed in the future.
#[unstable(feature = "allocator_ext", issue = "163177", implied_by = "allocator_api")]
#[rustc_const_unstable(feature = "const_heap", issue = "79597")]
#[unstable_feature_bound(allocator_ext)]
const unsafe impl<P> Allocator for core::pin::Pin<P>
where
    P: [const] core::ops::Deref<Target: [const] Allocator> + core::pin::PinSafePointer,
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

/// This trait impl is unstable and only usable on nightly with `feature(allocator_ext)`.
#[unstable(feature = "allocator_ext", issue = "163177", implied_by = "allocator_api")]
#[unstable_feature_bound(allocator_ext)]
impl<P> AllocatorNightly for core::pin::Pin<P> where
    P: core::ops::Deref<Target: Allocator> + core::pin::PinSafePointer
{
}

#[unstable(feature = "allocator_ext", issue = "163177", implied_by = "allocator_api")]
unsafe impl<A: Allocator + ?Sized> AllocatorClone for &A {}

// If an allocator is `StaticAllocator` all equivalent allocators must also uphold
// its semantics, and references are equivalent to the allocator they reference.
#[unstable(feature = "allocator_ext", issue = "163177", implied_by = "allocator_api")]
unsafe impl<A: StaticAllocator + ?Sized> StaticAllocator for &A {}

#[unstable(feature = "allocator_ext", issue = "163177", implied_by = "allocator_api")]
unsafe impl<A: StaticAllocator + ?Sized> StaticAllocator for &mut A {}

/// Allocator marker trait that is only implemented stably for `Global`.
///
/// Enabling `feature(allocator_ext)` causes this trait to be implemented for all
/// allocators in std, and for any downstream allocator that explicitly implements it.
///
/// This trait is used as a bound whenever a function constructing
/// a type with an `#[unstable] A: Allocator = Global` parameter
/// may be callable for `A != Global`.
///
/// This is to prevent stable code from e.g. constructing `Arc<T, NotGlobal>`
/// using the `From<Box<T, A>> for Arc<T, A>` impl.
///
/// Note that this trait *will* be removed in the future, but downstream
/// allocators should still feel free to implement it on nightly, to
/// allow their allocator to be used with all unstable allocator APIs.
#[unstable(feature = "allocator_ext", issue = "163177", implied_by = "allocator_api")]
pub trait AllocatorNightly: Allocator {}

/// This trait impl is unstable and only usable on nightly with `feature(allocator_ext)`.
#[unstable(feature = "allocator_ext", issue = "163177", implied_by = "allocator_api")]
#[unstable_feature_bound(allocator_ext)]
impl<A: Allocator + ?Sized> AllocatorNightly for &A {}

/// This trait impl is unstable and only usable on nightly with `feature(allocator_ext)`.
#[unstable(feature = "allocator_ext", issue = "163177", implied_by = "allocator_api")]
#[unstable_feature_bound(allocator_ext)]
impl<A: Allocator + ?Sized> AllocatorNightly for &mut A {}
