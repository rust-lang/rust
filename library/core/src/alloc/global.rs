use super::{AllocError, Allocator, GlobalAllocator, Layout};
use crate::mem::Alignment;
use crate::ptr::NonNull;
use crate::{cmp, hint, intrinsics, ptr};

/// A memory allocator that can be registered as the standard library’s default
/// through the `#[global_allocator]` attribute.
///
/// Some of the methods require that a memory block be *currently
/// allocated* via an allocator. This means that:
///
/// * the starting address for that memory block was previously
///   returned by a previous call to an allocation method
///   such as `alloc`, and
///
/// * the memory block has not been subsequently deallocated, where
///   blocks are deallocated either by being passed to a deallocation
///   method such as `dealloc` or by being
///   passed to a reallocation method that returns a non-null pointer.
///
/// # Example
///
/// ```standalone_crate
/// use std::alloc::{GlobalAlloc, Layout};
/// use std::cell::UnsafeCell;
/// use std::ptr::null_mut;
/// use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};
///
/// const ARENA_SIZE: usize = 128 * 1024;
/// const MAX_SUPPORTED_ALIGN: usize = 4096;
/// #[repr(C, align(4096))] // 4096 == MAX_SUPPORTED_ALIGN
/// struct SimpleAllocator {
///     arena: UnsafeCell<[u8; ARENA_SIZE]>,
///     remaining: AtomicUsize, // we allocate from the top, counting down
/// }
///
/// #[global_allocator]
/// static ALLOCATOR: SimpleAllocator = SimpleAllocator {
///     arena: UnsafeCell::new([0x55; ARENA_SIZE]),
///     remaining: AtomicUsize::new(ARENA_SIZE),
/// };
///
/// unsafe impl Sync for SimpleAllocator {}
///
/// unsafe impl GlobalAlloc for SimpleAllocator {
///     unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
///         let size = layout.size();
///         let align = layout.align();
///
///         // `Layout` contract forbids making a `Layout` with align=0, or align not power of 2.
///         // So we can safely use a mask to ensure alignment without worrying about UB.
///         let align_mask_to_round_down = !(align - 1);
///
///         if align > MAX_SUPPORTED_ALIGN {
///             return null_mut();
///         }
///
///         let mut allocated = 0;
///         if self
///             .remaining
///             .try_update(Relaxed, Relaxed, |mut remaining| {
///                 if size > remaining {
///                     return None;
///                 }
///                 remaining -= size;
///                 remaining &= align_mask_to_round_down;
///                 allocated = remaining;
///                 Some(remaining)
///             })
///             .is_err()
///         {
///             return null_mut();
///         };
///         unsafe { self.arena.get().cast::<u8>().add(allocated) }
///     }
///     unsafe fn dealloc(&self, _ptr: *mut u8, _layout: Layout) {}
/// }
///
/// fn main() {
///     let _s = format!("allocating a string!");
///     let currently = ALLOCATOR.remaining.load(Relaxed);
///     println!("allocated so far: {}", ARENA_SIZE - currently);
/// }
/// ```
///
/// # The `#[global_allocator]` attribute
///
/// As the example above demonstrates, the `#[global_allocator]` attribute can be used to register a
/// concrete `static` of a type that implements this trait to become *the* global allocator
/// for the current program. That global allocator can be invoked via the functions [`alloc`],
/// [`alloc_zeroed`], [`dealloc`], [`realloc`]). Note, however, that invoking those functions is
/// *not* equivalent to directly invoking the underlying methods on the declared global allocator!
/// Users of the global allocator cannot assume anything about what the allocator does (even if they know which allocator is being used),
/// and implementors of the allocator cannot assume anything about what the program does (even if they know how the allocator is being used).
/// Both can only assume the documented requirements for the respective other party of this contract.
/// This means:
///
/// - Allocation functions may non-deterministically entirely skip the underlying allocator, e.g. if the
///   compiler can show that this allocation can be replaced by a stack variable. The compiler may
///   also merge multiple allocation operations into one, as long as it can also adjust all
///   corresponding deallocation operations accordingly.
/// - An allocation created by invoking [`alloc`], [`alloc_zeroed`], or [`realloc`] has exactly the
///   size and minimum alignment defined by `layout`, even if the underlying allocator makes
///   stronger promises.
/// - An allocation created by invoking [`alloc`], [`alloc_zeroed`], or [`realloc`] can only be
///   freed by invoking [`dealloc`] or [`realloc`]. In particular, passing a pointer to such an
///   allocation directly to the underlying method on [`GlobalAlloc`] is not permitted. Until one of
///   those functions is called, it is undefined behavior to access the memory that backs this
///   allocation with any pointer not derived from the return value of this function (e.g., with
///   internal pointers the allocator might keep around).
/// - The pointer passed to [`dealloc`] or [`realloc`] must have been obtained by invoking [`alloc`],
///   [`alloc_zeroed`], or [`realloc`]. In particular, passing a pointer returned by the underlying
///   methods on [`GlobalAlloc`] is not permitted.
/// - [`alloc`] de-initializes the contents of the allocation before handing it to the user. So even
///   if you control the underlying allocator and know that it explicitly initialized this memory,
///   you cannot rely on it being initialized. For a [`realloc`] that grows an allocation, this
///   applies to the newly allocated part.
/// - [`dealloc`] de-initializes the contents of the allocation before handing it to the allocator.
///   So even if you know that the program previously initialized that memory, the allocator cannot
///   rely on it being initialized. For a [`realloc`] that shrinks an allocation, this applies to
///   the part being removed.
///
/// [`alloc`]: ../../std/alloc/fn.alloc.html
/// [`alloc_zeroed`]: ../../std/alloc/fn.alloc_zeroed.html
/// [`dealloc`]: ../../std/alloc/fn.dealloc.html
/// [`realloc`]: ../../std/alloc/fn.realloc.html
///
/// The first point means that you cannot rely on global allocations actually happening, even if
/// there are explicit global allocations in the source. The optimizer may detect unused global
/// allocations that it can either eliminate entirely or move to the stack and thus never invoke the
/// global allocator. The optimizer may further assume that allocation is infallible, so code that
/// used to fail due to allocator failures may now suddenly work because the optimizer worked around
/// the need for an allocation. More concretely, the following code example is unsound, irrespective
/// of whether your custom allocator allows counting how many allocations have happened.
///
/// ```rust,ignore (unsound and has placeholders)
/// drop(Box::new(42));
/// let number_of_heap_allocs = /* call private allocator API */;
/// unsafe { std::hint::assert_unchecked(number_of_heap_allocs > 0); }
/// ```
///
/// Note that the optimizations mentioned above are not the only
/// optimization that can be applied. You may generally not rely on global allocations
/// happening if they can be removed without changing program behavior.
/// Whether allocations happen or not is not part of the program behavior, even if it
/// could be detected via an allocator that tracks allocations by printing or otherwise
/// having side effects.
///
/// # Safety
///
/// The `GlobalAlloc` trait is an `unsafe` trait for a number of reasons, and
/// implementors must ensure that they adhere to these contracts:
///
/// * It is undefined behavior for the allocator to read, write, or deallocate any memory that
///   is *currently allocated*. This memory is owned by the user, the allocator must not touch it.
///
/// * It's undefined behavior if global allocators unwind. This restriction may
///   be lifted in the future, but currently a panic from any of these
///   functions may lead to memory unsafety.
///
/// * Callers of this trait are allowed to rely on the contracts defined on each method, and
///   implementors must ensure such contracts remain true.
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
///    [`Clone`] implementation.
///
/// [`std`]: ../../std/index.html
/// [`std::sync::Mutex`]: ../../std/sync/struct.Mutex.html
/// [`std::thread_local`]: ../../std/macro.thread_local.html
/// [`std::thread::current`]: ../../std/thread/fn.current.html
/// [`std::thread::park`]: ../../std/thread/fn.park.html
/// [`std::thread::Thread`]: ../../std/thread/struct.Thread.html
/// [`unpark`]: ../../std/thread/struct.Thread.html#method.unpark

#[stable(feature = "global_alloc", since = "1.28.0")]
pub unsafe trait GlobalAlloc {
    /// Allocates memory as described by the given `layout`.
    ///
    /// Returns a pointer to newly-allocated memory,
    /// or null to indicate allocation failure.
    ///
    /// # Safety
    ///
    /// `layout` must have non-zero size. Attempting to allocate for a zero-sized `layout` will
    /// result in undefined behavior.
    ///
    /// (Extension subtraits might provide more specific bounds on
    /// behavior, e.g., guarantee a sentinel address or a null pointer
    /// in response to a zero-size allocation request.)
    ///
    /// The allocated block of memory may or may not be initialized.
    ///
    /// # Errors
    ///
    /// Returning a null pointer indicates that either memory is exhausted
    /// or `layout` does not meet this allocator's size or alignment constraints.
    ///
    /// Implementations are encouraged to return null on memory
    /// exhaustion rather than aborting, but this is not
    /// a strict requirement. (Specifically: it is *legal* to
    /// implement this trait atop an underlying native allocation
    /// library that aborts on memory exhaustion.)
    ///
    /// Clients wishing to abort computation in response to an
    /// allocation error are encouraged to call the [`handle_alloc_error`] function,
    /// rather than directly invoking `panic!` or similar (but note that both may unwind).
    ///
    /// [`handle_alloc_error`]: ../../alloc/alloc/fn.handle_alloc_error.html
    #[stable(feature = "global_alloc", since = "1.28.0")]
    unsafe fn alloc(&self, layout: Layout) -> *mut u8;

    /// Deallocates the block of memory at the given `ptr` pointer with the given `layout`.
    ///
    /// # Safety
    ///
    /// The caller must ensure:
    ///
    /// * `ptr` is a block of memory currently allocated via this allocator and,
    ///
    /// * `layout` is the same layout that was used to allocate that block of
    ///   memory.
    ///
    /// Otherwise the behavior is undefined.
    #[stable(feature = "global_alloc", since = "1.28.0")]
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout);

    /// Behaves like `alloc`, but also ensures that the contents
    /// are set to zero before being returned.
    ///
    /// # Safety
    ///
    /// The caller has to ensure that `layout` has non-zero size. Like `alloc`
    /// zero sized `layout` will result in undefined behavior.
    /// However the allocated block of memory is guaranteed to be initialized.
    ///
    /// # Errors
    ///
    /// Returning a null pointer indicates that either memory is exhausted
    /// or `layout` does not meet allocator's size or alignment constraints,
    /// just as in `alloc`.
    ///
    /// Clients wishing to abort computation in response to an
    /// allocation error are encouraged to call the [`handle_alloc_error`] function,
    /// rather than directly invoking `panic!` or similar.
    ///
    /// [`handle_alloc_error`]: ../../alloc/alloc/fn.handle_alloc_error.html
    #[stable(feature = "global_alloc", since = "1.28.0")]
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let size = layout.size();
        // SAFETY: the safety contract for `alloc` must be upheld by the caller.
        let ptr = unsafe { self.alloc(layout) };
        if !ptr.is_null() {
            // SAFETY: as allocation succeeded, the region from `ptr`
            // of size `size` is guaranteed to be valid for writes.
            unsafe { ptr::write_bytes(ptr, 0, size) };
        }
        ptr
    }

    /// Shrinks or grows a block of memory to the given `new_size` in bytes.
    /// The block is described by the given `ptr` pointer and `layout`.
    ///
    /// If this returns a non-null pointer, then ownership of the memory block
    /// referenced by `ptr` has been transferred to this allocator.
    /// Any access to the old `ptr` is Undefined Behavior, even if the
    /// allocation remained in-place. The newly returned pointer is the only valid pointer
    /// for accessing this memory now.
    ///
    /// The new memory block is allocated with `layout`,
    /// but with the `size` updated to `new_size` in bytes.
    /// This new layout must be used when deallocating the new memory block with `dealloc`.
    /// The range `0..min(layout.size(), new_size)` of the new memory block is
    /// guaranteed to have the same values as the original block.
    ///
    /// If this method returns null, then ownership of the memory
    /// block has not been transferred to this allocator, and the
    /// contents of the memory block are unaltered.
    ///
    /// # Safety
    ///
    /// The caller must ensure that:
    ///
    /// * `ptr` is allocated via this allocator,
    ///
    /// * `layout` is the same layout that was used
    ///   to allocate that block of memory,
    ///
    /// * `new_size` is greater than zero.
    ///
    /// * `new_size`, when rounded up to the nearest multiple of `layout.align()`,
    ///   does not overflow `isize` (i.e., the rounded value must be less than or
    ///   equal to `isize::MAX`).
    ///
    /// If these are not followed, the behavior is undefined.
    ///
    /// (Extension subtraits might provide more specific bounds on
    /// behavior, e.g., guarantee a sentinel address or a null pointer
    /// in response to a zero-size allocation request.)
    ///
    /// # Errors
    ///
    /// Returns null if the new layout does not meet the size
    /// and alignment constraints of the allocator, or if reallocation
    /// otherwise fails.
    ///
    /// Implementations are encouraged to return null on memory
    /// exhaustion rather than panicking or aborting, but this is not
    /// a strict requirement. (Specifically: it is *legal* to
    /// implement this trait atop an underlying native allocation
    /// library that aborts on memory exhaustion.)
    ///
    /// Clients wishing to abort computation in response to a
    /// reallocation error are encouraged to call the [`handle_alloc_error`] function,
    /// rather than directly invoking `panic!` or similar.
    ///
    /// [`handle_alloc_error`]: ../../alloc/alloc/fn.handle_alloc_error.html
    #[stable(feature = "global_alloc", since = "1.28.0")]
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let alignment = layout.alignment();
        // SAFETY: the caller must ensure that the `new_size` does not overflow
        // when rounded up to the next multiple of `alignment`.
        let new_layout = unsafe { Layout::from_size_alignment_unchecked(new_size, alignment) };
        // SAFETY: the caller must ensure that `new_layout` is greater than zero.
        let new_ptr = unsafe { self.alloc(new_layout) };
        if !new_ptr.is_null() {
            // SAFETY: the previously allocated block cannot overlap the newly allocated block.
            // The safety contract for `dealloc` must be upheld by the caller.
            unsafe {
                ptr::copy_nonoverlapping(ptr, new_ptr, cmp::min(layout.size(), new_size));
                self.dealloc(ptr, layout);
            }
        }
        new_ptr
    }
}

/// Allows all [`GlobalAllocator`]s to be used with the legacy [`GlobalAlloc`] interface.
#[stable(feature = "global_alloc", since = "1.28.0")]
unsafe impl<A> GlobalAlloc for A
where
    A: GlobalAllocator + ?Sized,
{
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: guaranteed by the caller.
        // This might lead to the removal of zero-size checks inside the
        // `Allocator` implementation.
        unsafe { hint::assert_unchecked(layout.size() != 0) };
        match self.allocate(layout) {
            Ok(ptr) => ptr.cast().as_ptr(),
            Err(AllocError) => ptr::null_mut(),
        }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: guaranteed by the caller.
        unsafe { hint::assert_unchecked(layout.size() != 0) };
        // SAFETY: only non-null pointers can be currently allocated.
        let ptr = unsafe { NonNull::new_unchecked(ptr) };
        // SAFETY: guaranteed by caller.
        unsafe { self.deallocate(ptr, layout) };
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        // SAFETY: guaranteed by the caller.
        unsafe { hint::assert_unchecked(layout.size() != 0) };
        match self.allocate_zeroed(layout) {
            Ok(ptr) => ptr.cast().as_ptr(),
            Err(AllocError) => ptr::null_mut(),
        }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        // SAFETY: guaranteed by the caller.
        unsafe { hint::assert_unchecked(layout.size() != 0) };
        // SAFETY: guaranteed by the caller.
        unsafe { hint::assert_unchecked(new_size != 0) };

        // SAFETY: only non-null pointers can be currently allocated.
        let ptr = unsafe { NonNull::new_unchecked(ptr) };
        let alignment = layout.alignment();
        // SAFETY: the caller must ensure that the `new_size` does not overflow
        // when rounded up to the next multiple of `alignment`.
        let new_layout = unsafe { Layout::from_size_alignment_unchecked(new_size, alignment) };

        // SAFETY:
        // Two preconditions are guaranteed by the caller:
        // * `ptr` is currently allocated with this allocator.
        // * `layout` fits the block of memory.
        // The size precondition is upheld by selecting between `grow` and `shrink`
        // based on the size.
        let ptr = unsafe {
            if new_size >= layout.size() {
                self.grow(ptr, layout, new_layout)
            } else {
                self.shrink(ptr, layout, new_layout)
            }
        };

        match ptr {
            Ok(ptr) => ptr.cast().as_ptr(),
            Err(AllocError) => ptr::null_mut(),
        }
    }
}

unsafe extern "Rust" {
    // These are the magic symbols to call the global allocator. rustc generates
    // them to call the global allocator if there is a `#[global_allocator]` attribute
    // (the code expanding that attribute macro generates those functions), or to call
    // the default implementations in std (`__rdl_alloc` etc. in `library/std/src/alloc.rs`)
    // otherwise.
    #[rustc_allocator]
    #[rustc_nounwind]
    #[rustc_std_internal_symbol]
    #[rustc_allocator_zeroed_variant = "__rust_alloc_zeroed"]
    fn __rust_alloc(size: usize, align: Alignment) -> *mut u8;
    #[rustc_deallocator]
    #[rustc_nounwind]
    #[rustc_std_internal_symbol]
    fn __rust_dealloc(ptr: NonNull<u8>, size: usize, align: Alignment);
    #[rustc_reallocator]
    #[rustc_nounwind]
    #[rustc_std_internal_symbol]
    fn __rust_realloc(
        ptr: NonNull<u8>,
        old_size: usize,
        align: Alignment,
        new_size: usize,
    ) -> *mut u8;
    #[rustc_allocator_zeroed]
    #[rustc_nounwind]
    #[rustc_std_internal_symbol]
    fn __rust_alloc_zeroed(size: usize, align: Alignment) -> *mut u8;

    #[rustc_nounwind]
    #[rustc_std_internal_symbol]
    fn __rust_no_alloc_shim_is_unstable_v2();
}

/// The global memory allocator.
///
/// This type implements the [`Allocator`] trait by forwarding calls
/// to the allocator registered with the `#[global_allocator]` attribute
/// if there is one, or the `std` crate’s default.
///
/// Note: while this type is unstable, the functionality it provides can be
/// accessed through the [free functions in `alloc`](super#functions).
#[unstable(feature = "allocator_api", issue = "32838")]
#[derive(Copy, Debug)]
#[derive_const(Clone, Default)]
// the compiler needs to know when a Box uses the global allocator vs a custom one
#[lang = "global_alloc_ty"]
pub struct Global;

/// Allocates memory with the global allocator.
///
/// This function forwards calls to the [`GlobalAlloc::alloc`] method
/// of the allocator registered with the `#[global_allocator]` attribute
/// if there is one, or the `std` crate’s default.
///
/// Note, however, that invoking this function is *not* equivalent to invoking the underlying
/// [`GlobalAlloc::alloc`] method of the registered allocator directly. Users of this function
/// cannot assume anything about what the allocator does, other than the documented requirements.
/// This means:
///
/// - This function may non-deterministically entirely skip the underlying allocator, e.g. if the
///   compiler can show that this allocation can be replaced by a stack variable. The compiler may
///   also merge multiple allocation operations into one, as long as it can also adjust all
///   corresponding deallocation operations accordingly.
/// - An allocation created by invoking this function has exactly the size and minimum alignment
///   defined by `layout`, even if the underlying allocator makes stronger promises.
/// - The allocation can only be freed by invoking [`dealloc`] or [`realloc`]. In particular,
///   passing a pointer to such an allocation directly to the underlying method on [`GlobalAlloc`] is
///   not permitted. Until one of those functions is called, it is undefined behavior to access the
///   memory that backs this allocation with any pointer not derived from the return value of this
///   function (e.g., with internal pointers the allocator might keep around).
/// - This function de-initializes the contents of the allocation before handing it to the user. So even
///   if you control the underlying allocator and know that it explicitly initialized this memory,
///   you cannot rely on it being initialized.
///
/// Users of this function have to consider that in the future, allocators may be allowed to unwind.
///
/// This function is expected to be deprecated in favor of the `allocate` method
/// of the [`Global`] type when it and the [`Allocator`] trait become stable.
///
/// # Safety
///
/// See [`GlobalAlloc::alloc`].
///
/// # Examples
///
/// ```
/// use std::alloc::{alloc, dealloc, handle_alloc_error, Layout};
///
/// unsafe {
///     let layout = Layout::new::<u16>();
///     let ptr = alloc(layout);
///     if ptr.is_null() {
///         handle_alloc_error(layout);
///     }
///
///     *(ptr as *mut u16) = 42;
///     assert_eq!(*(ptr as *mut u16), 42);
///
///     dealloc(ptr, layout);
/// }
/// ```
#[unstable(feature = "core_global_alloc", issue = "none")]
#[must_use = "losing the pointer will leak memory"]
#[inline]
#[cfg_attr(miri, track_caller)] // even without panics, this helps for Miri backtraces
pub unsafe fn alloc(layout: Layout) -> *mut u8 {
    // SAFETY: These shims have the same requirements as the parent method.
    unsafe {
        // Make sure we don't accidentally allow omitting the allocator shim in
        // stable code until it is actually stabilized.
        __rust_no_alloc_shim_is_unstable_v2();

        __rust_alloc(layout.size(), layout.alignment())
    }
}

/// Deallocates memory with the global allocator.
///
/// This function forwards calls to the [`GlobalAlloc::dealloc`] method
/// of the allocator registered with the `#[global_allocator]` attribute
/// if there is one, or the `std` crate’s default.
///
/// Note, however, that invoking this function is *not* equivalent to invoking the underlying
/// [`GlobalAlloc::dealloc`] method of the registered allocator directly. Users of this function
/// cannot assume anything about what the allocator does, other than the documented requirements.
/// This means:
///
/// - This function may non-deterministically entirely skip the underlying allocator, e.g. if the
///   compiler can show that this allocation can be replaced by a stack variable. The compiler may
///   also merge multiple allocation operations into one, as long as it can also adjust all
///   corresponding deallocation operations accordingly.
/// - The pointer passed to this function must have been obtained by invoking [`alloc`],
///   [`alloc_zeroed`], or [`realloc`]. In particular, passing a pointer returned by the underlying
///   methods on [`GlobalAlloc`] is not permitted.
/// - This function de-initializes the contents of the allocation before handing it to the allocator.
///   So even if you know that the program previously initialized that memory, the allocator cannot
///   rely on it being initialized.
///
/// Users of this function have to consider that in the future, allocators may be allowed to unwind.
///
/// This function is expected to be deprecated in favor of the `deallocate` method
/// of the [`Global`] type when it and the [`Allocator`] trait become stable.
///
/// # Safety
///
/// See [`GlobalAlloc::dealloc`].
#[unstable(feature = "core_global_alloc", issue = "none")]
#[inline]
#[cfg_attr(miri, track_caller)] // even without panics, this helps for Miri backtraces
pub unsafe fn dealloc(ptr: *mut u8, layout: Layout) {
    // SAFETY: These shims have the same requirements as the parent method.
    unsafe { dealloc_nonnull(NonNull::new_unchecked(ptr), layout) }
}

/// Same as [`dealloc`] but when you already have a non-null pointer
#[inline]
#[cfg_attr(miri, track_caller)] // even without panics, this helps for Miri backtraces
unsafe fn dealloc_nonnull(ptr: NonNull<u8>, layout: Layout) {
    // SAFETY: These shims have the same requirements as the parent method.
    unsafe { __rust_dealloc(ptr, layout.size(), layout.alignment()) }
}

/// Reallocates memory with the global allocator.
///
/// This function forwards calls to the [`GlobalAlloc::realloc`] method
/// of the allocator registered with the `#[global_allocator]` attribute
/// if there is one, or the `std` crate’s default.
///
/// Note, however, that invoking this function is *not* equivalent to invoking the underlying
/// [`GlobalAlloc::realloc`] method of the registered allocator directly. Users of this function
/// cannot assume anything about what the allocator does, other than the documented requirements.
/// This means:
///
/// - This function may non-deterministically entirely skip the underlying allocator, e.g. if the
///   compiler can show that this allocation can be replaced by a stack variable. The compiler may
///   also merge multiple allocation operations into one, as long as it can also adjust all
///   corresponding deallocation operations accordingly.
/// - The pointer passed to this function must have been obtained by invoking [`alloc`],
///   [`alloc_zeroed`], or [`realloc`]. In particular, passing a pointer returned by the underlying
///   methods on [`GlobalAlloc`] is not permitted.
/// - An allocation created by invoking this function has exactly the size and minimum alignment
///   defined by `layout`, even if the underlying allocator makes stronger promises.
/// - The allocation can only be freed by invoking [`dealloc`] or [`realloc`]. In particular,
///   passing a pointer to such an allocation directly to the underlying method on [`GlobalAlloc`] is
///   not permitted. Until one of those functions is called, it is undefined behavior to access the
///   memory that backs this allocation with any pointer not derived from the return value of this
///   function (e.g., with internal pointers the allocator might keep around).
/// - If this grows the allocation, the contents of the grown part of the new allocation allocation
///   are de-initialized by this function before returning.
/// - If this shrinks the allocation, the contents of the removed part of the old allocation are
///   de-initialized by this function before invoking the underlying allocator.
///
/// Users of this function have to consider that in the future, allocators may be allowed to unwind.
///
/// This function is expected to be deprecated in favor of the `grow` and `shrink` methods
/// of the [`Global`] type when it and the [`Allocator`] trait become stable.
///
/// # Safety
///
/// See [`GlobalAlloc::realloc`].
#[unstable(feature = "core_global_alloc", issue = "none")]
#[must_use = "losing the pointer will leak memory"]
#[inline]
#[cfg_attr(miri, track_caller)] // even without panics, this helps for Miri backtraces
pub unsafe fn realloc(ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
    // SAFETY: These shims have the same requirements as the parent method.
    unsafe { realloc_nonnull(NonNull::new_unchecked(ptr), layout, new_size) }
}

/// Same as [`realloc`] but when you already have a non-null pointer
#[inline]
#[cfg_attr(miri, track_caller)] // even without panics, this helps for Miri backtraces
unsafe fn realloc_nonnull(ptr: NonNull<u8>, layout: Layout, new_size: usize) -> *mut u8 {
    // SAFETY: These shims have the same requirements as the parent method.
    unsafe { __rust_realloc(ptr, layout.size(), layout.alignment(), new_size) }
}

/// Allocates zero-initialized memory with the global allocator.
///
/// This function forwards calls to the [`GlobalAlloc::alloc_zeroed`] method
/// of the allocator registered with the `#[global_allocator]` attribute
/// if there is one, or the `std` crate’s default.
///
/// Note, however, that invoking this function is *not* equivalent to invoking the underlying
/// [`GlobalAlloc::alloc_zeroed`] method of the registered allocator directly. Users of this
/// function cannot assume anything about what the allocator does, other than the documented
/// requirements. This means:
///
/// - This function may non-deterministically entirely skip the underlying allocator, e.g. if the
///   compiler can show that this allocation can be replaced by a stack variable. The compiler may
///   also merge multiple allocation operations into one, as long as it can also adjust all
///   corresponding deallocation operations accordingly.
/// - The allocation can only be freed by invoking [`dealloc`] or [`realloc`]. In particular,
///   passing a pointer to such an allocation directly to the underlying method on [`GlobalAlloc`] is
///   not permitted. Until one of those functions is called, it is undefined behavior to access the
///   memory that backs this allocation with any pointer not derived from the return value of this
///   function (e.g., with internal pointers the allocator might keep around).
/// - An allocation created by invoking this function has exactly the size and minimum alignment
///   defined by `layout`, even if the underlying allocator makes stronger promises.
///
/// Users of this function have to consider that in the future, allocators may be allowed to unwind.
///
/// This function is expected to be deprecated in favor of the `allocate_zeroed` method
/// of the [`Global`] type when it and the [`Allocator`] trait become stable.
///
/// # Safety
///
/// See [`GlobalAlloc::alloc_zeroed`].
///
/// # Examples
///
/// ```
/// use std::alloc::{alloc_zeroed, dealloc, handle_alloc_error, Layout};
///
/// unsafe {
///     let layout = Layout::new::<u16>();
///     let ptr = alloc_zeroed(layout);
///     if ptr.is_null() {
///         handle_alloc_error(layout);
///     }
///
///     assert_eq!(*(ptr as *mut u16), 0);
///
///     dealloc(ptr, layout);
/// }
/// ```
#[unstable(feature = "core_global_alloc", issue = "none")]
#[must_use = "losing the pointer will leak memory"]
#[inline]
#[cfg_attr(miri, track_caller)] // even without panics, this helps for Miri backtraces
pub unsafe fn alloc_zeroed(layout: Layout) -> *mut u8 {
    // SAFETY: These shims have the same requirements as the parent method.
    unsafe {
        // Make sure we don't accidentally allow omitting the allocator shim in
        // stable code until it is actually stabilized.
        __rust_no_alloc_shim_is_unstable_v2();

        __rust_alloc_zeroed(layout.size(), layout.alignment())
    }
}

impl Global {
    #[inline]
    #[cfg_attr(miri, track_caller)] // even without panics, this helps for Miri backtraces
    fn alloc_impl_runtime(layout: Layout, zeroed: bool) -> Result<NonNull<[u8]>, AllocError> {
        match layout.size() {
            0 => Ok(layout.dangling_ptr().cast_slice(0)),
            // SAFETY: `layout` is non-zero in size,
            size => unsafe {
                let raw_ptr = if zeroed { alloc_zeroed(layout) } else { alloc(layout) };
                let ptr = NonNull::new(raw_ptr).ok_or(AllocError)?;
                Ok(ptr.cast_slice(size))
            },
        }
    }

    #[inline]
    #[cfg_attr(miri, track_caller)] // even without panics, this helps for Miri backtraces
    fn deallocate_impl_runtime(ptr: NonNull<u8>, layout: Layout) {
        if layout.size() != 0 {
            // SAFETY:
            // * We have checked that `layout` is non-zero in size.
            // * The caller is obligated to provide a layout that "fits", and in this case,
            //   "fit" always means a layout that is equal to the original, because our
            //   `allocate()`, `grow()`, and `shrink()` implementations never returns a larger
            //   allocation than requested.
            // * Other conditions must be upheld by the caller, as per `Allocator::deallocate()`'s
            //   safety documentation.
            unsafe { dealloc_nonnull(ptr, layout) }
        }
    }

    // SAFETY: Same as `Allocator::grow`
    #[inline]
    #[cfg_attr(miri, track_caller)] // even without panics, this helps for Miri backtraces
    fn grow_impl_runtime(
        &self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
        zeroed: bool,
    ) -> Result<NonNull<[u8]>, AllocError> {
        debug_assert!(
            new_layout.size() >= old_layout.size(),
            "`new_layout.size()` must be greater than or equal to `old_layout.size()`"
        );

        match old_layout.size() {
            0 => self.alloc_impl(new_layout, zeroed),

            // SAFETY: `new_size` is non-zero as `old_size` is greater than or equal to `new_size`
            // as required by safety conditions. Other conditions must be upheld by the caller
            old_size if old_layout.align() == new_layout.align() => unsafe {
                let new_size = new_layout.size();

                // `realloc` probably checks for `new_size >= old_layout.size()` or something similar.
                hint::assert_unchecked(new_size >= old_layout.size());

                let raw_ptr = realloc_nonnull(ptr, old_layout, new_size);
                let ptr = NonNull::new(raw_ptr).ok_or(AllocError)?;
                if zeroed {
                    raw_ptr.add(old_size).write_bytes(0, new_size - old_size);
                }
                Ok(ptr.cast_slice(new_size))
            },

            // SAFETY: because `new_layout.size()` must be greater than or equal to `old_size`,
            // both the old and new memory allocation are valid for reads and writes for `old_size`
            // bytes. Also, because the old allocation wasn't yet deallocated, it cannot overlap
            // `new_ptr`. Thus, the call to `copy_nonoverlapping` is safe. The safety contract
            // for `dealloc` must be upheld by the caller.
            old_size => unsafe {
                let new_ptr = self.alloc_impl(new_layout, zeroed)?;
                ptr::copy_nonoverlapping(ptr.as_ptr(), new_ptr.as_mut_ptr(), old_size);
                self.deallocate(ptr, old_layout);
                Ok(new_ptr)
            },
        }
    }

    // SAFETY: Same as `Allocator::grow`
    #[inline]
    #[cfg_attr(miri, track_caller)] // even without panics, this helps for Miri backtraces
    fn shrink_impl_runtime(
        &self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
        _zeroed: bool,
    ) -> Result<NonNull<[u8]>, AllocError> {
        debug_assert!(
            new_layout.size() <= old_layout.size(),
            "`new_layout.size()` must be smaller than or equal to `old_layout.size()`"
        );

        match new_layout.size() {
            // SAFETY: conditions must be upheld by the caller
            0 => unsafe {
                self.deallocate(ptr, old_layout);
                Ok(new_layout.dangling_ptr().cast_slice(0))
            },

            // SAFETY: `new_size` is non-zero. Other conditions must be upheld by the caller
            new_size if old_layout.align() == new_layout.align() => unsafe {
                // `realloc` probably checks for `new_size <= old_layout.size()` or something similar.
                hint::assert_unchecked(new_size <= old_layout.size());

                let raw_ptr = realloc_nonnull(ptr, old_layout, new_size);
                let ptr = NonNull::new(raw_ptr).ok_or(AllocError)?;
                Ok(ptr.cast_slice(new_size))
            },

            // SAFETY: because `new_size` must be smaller than or equal to `old_layout.size()`,
            // both the old and new memory allocation are valid for reads and writes for `new_size`
            // bytes. Also, because the old allocation wasn't yet deallocated, it cannot overlap
            // `new_ptr`. Thus, the call to `copy_nonoverlapping` is safe. The safety contract
            // for `dealloc` must be upheld by the caller.
            new_size => unsafe {
                let new_ptr = self.allocate(new_layout)?;
                ptr::copy_nonoverlapping(ptr.as_ptr(), new_ptr.as_mut_ptr(), new_size);
                self.deallocate(ptr, old_layout);
                Ok(new_ptr)
            },
        }
    }

    // SAFETY: Same as `Allocator::allocate`
    #[inline]
    #[cfg_attr(miri, track_caller)] // even without panics, this helps for Miri backtraces
    #[rustc_const_unstable(feature = "const_heap", issue = "79597")]
    const fn alloc_impl(&self, layout: Layout, zeroed: bool) -> Result<NonNull<[u8]>, AllocError> {
        intrinsics::const_eval_select(
            (layout, zeroed),
            Global::alloc_impl_const,
            Global::alloc_impl_runtime,
        )
    }

    // SAFETY: Same as `Allocator::deallocate`
    #[inline]
    #[cfg_attr(miri, track_caller)] // even without panics, this helps for Miri backtraces
    #[rustc_const_unstable(feature = "const_heap", issue = "79597")]
    const unsafe fn deallocate_impl(&self, ptr: NonNull<u8>, layout: Layout) {
        intrinsics::const_eval_select(
            (ptr, layout),
            Global::deallocate_impl_const,
            Global::deallocate_impl_runtime,
        )
    }

    // SAFETY: Same as `Allocator::grow`
    #[inline]
    #[cfg_attr(miri, track_caller)] // even without panics, this helps for Miri backtraces
    #[rustc_const_unstable(feature = "const_heap", issue = "79597")]
    const unsafe fn grow_impl(
        &self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
        zeroed: bool,
    ) -> Result<NonNull<[u8]>, AllocError> {
        intrinsics::const_eval_select(
            (self, ptr, old_layout, new_layout, zeroed),
            Global::grow_shrink_impl_const,
            Global::grow_impl_runtime,
        )
    }

    // SAFETY: Same as `Allocator::shrink`
    #[inline]
    #[cfg_attr(miri, track_caller)] // even without panics, this helps for Miri backtraces
    #[rustc_const_unstable(feature = "const_heap", issue = "79597")]
    const unsafe fn shrink_impl(
        &self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        intrinsics::const_eval_select(
            (self, ptr, old_layout, new_layout, false),
            Global::grow_shrink_impl_const,
            Global::shrink_impl_runtime,
        )
    }

    #[inline]
    #[cfg_attr(miri, track_caller)] // even without panics, this helps for Miri backtraces
    #[rustc_const_unstable(feature = "const_heap", issue = "79597")]
    const fn alloc_impl_const(layout: Layout, zeroed: bool) -> Result<NonNull<[u8]>, AllocError> {
        match layout.size() {
            0 => Ok(layout.dangling_ptr().cast_slice(0)),
            // SAFETY: `layout` is non-zero in size,
            size => unsafe {
                let raw_ptr = intrinsics::const_allocate(layout.size(), layout.align());
                let ptr = NonNull::new(raw_ptr).ok_or(AllocError)?;
                if zeroed {
                    // SAFETY: the pointer returned by `const_allocate` is valid to write to.
                    ptr.write_bytes(0, size);
                }
                Ok(ptr.cast_slice(size))
            },
        }
    }

    #[inline]
    #[cfg_attr(miri, track_caller)] // even without panics, this helps for Miri backtraces
    #[rustc_const_unstable(feature = "const_heap", issue = "79597")]
    const fn deallocate_impl_const(ptr: NonNull<u8>, layout: Layout) {
        if layout.size() != 0 {
            // SAFETY: We checked for nonzero size; other preconditions must be upheld by caller.
            unsafe {
                intrinsics::const_deallocate(ptr.as_ptr(), layout.size(), layout.align());
            }
        }
    }

    #[inline]
    #[cfg_attr(miri, track_caller)] // even without panics, this helps for Miri backtraces
    #[rustc_const_unstable(feature = "const_heap", issue = "79597")]
    const fn grow_shrink_impl_const(
        &self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
        zeroed: bool,
    ) -> Result<NonNull<[u8]>, AllocError> {
        let new_ptr = Global::alloc_impl_const(new_layout, zeroed)?;
        // SAFETY: both pointers are valid and this operations is in bounds.
        unsafe {
            ptr::copy_nonoverlapping(
                ptr.as_ptr(),
                new_ptr.as_mut_ptr(),
                cmp::min(old_layout.size(), new_layout.size()),
            );
        }
        Global::deallocate_impl_const(ptr, old_layout);
        Ok(new_ptr)
    }
}

#[unstable(feature = "allocator_api", issue = "32838")]
#[rustc_const_unstable(feature = "const_heap", issue = "79597")]
const unsafe impl Allocator for Global {
    #[inline]
    #[cfg_attr(miri, track_caller)] // even without panics, this helps for Miri backtraces
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        self.alloc_impl(layout, false)
    }

    #[inline]
    #[cfg_attr(miri, track_caller)] // even without panics, this helps for Miri backtraces
    fn allocate_zeroed(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        self.alloc_impl(layout, true)
    }

    #[inline]
    #[cfg_attr(miri, track_caller)] // even without panics, this helps for Miri backtraces
    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        // SAFETY: all conditions must be upheld by the caller
        unsafe { self.deallocate_impl(ptr, layout) }
    }

    #[inline]
    #[cfg_attr(miri, track_caller)] // even without panics, this helps for Miri backtraces
    unsafe fn grow(
        &self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        // SAFETY: all conditions must be upheld by the caller
        unsafe { self.grow_impl(ptr, old_layout, new_layout, false) }
    }

    #[inline]
    #[cfg_attr(miri, track_caller)] // even without panics, this helps for Miri backtraces
    unsafe fn grow_zeroed(
        &self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        // SAFETY: all conditions must be upheld by the caller
        unsafe { self.grow_impl(ptr, old_layout, new_layout, true) }
    }

    #[inline]
    #[cfg_attr(miri, track_caller)] // even without panics, this helps for Miri backtraces
    unsafe fn shrink(
        &self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        // SAFETY: all conditions must be upheld by the caller
        unsafe { self.shrink_impl(ptr, old_layout, new_layout) }
    }
}
