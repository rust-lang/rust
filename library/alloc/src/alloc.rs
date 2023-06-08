//! Memory allocation APIs

#![stable(feature = "alloc_module", since = "1.28.0")]

#[cfg(not(test))]
use core::intrinsics;
use core::intrinsics::{min_align_of_val, size_of_val};

use core::ptr::Unique;
#[cfg(not(test))]
use core::ptr::{self, NonNull};

#[stable(feature = "alloc_module", since = "1.28.0")]
#[doc(inline)]
pub use core::alloc::*;

#[cfg(test)]
mod tests;

extern "Rust" {
    // These are the magic symbols to call the global allocator. rustc generates
    // them to call `__rg_alloc` etc. if there is a `#[global_allocator]` attribute
    // (the code expanding that attribute macro generates those functions), or to call
    // the default implementations in std (`__rdl_alloc` etc. in `library/std/src/alloc.rs`)
    // otherwise.
    // The rustc fork of LLVM 14 and earlier also special-cases these function names to be able to optimize them
    // like `malloc`, `realloc`, and `free`, respectively.
    #[rustc_allocator]
    #[rustc_nounwind]
    fn __rust_alloc(size: usize, align: usize) -> *mut u8;
    #[rustc_deallocator]
    #[rustc_nounwind]
    fn __rust_dealloc(ptr: *mut u8, size: usize, align: usize);
    #[rustc_reallocator]
    #[rustc_nounwind]
    fn __rust_realloc(ptr: *mut u8, old_size: usize, align: usize, new_size: usize) -> *mut u8;
    #[rustc_allocator_zeroed]
    #[rustc_nounwind]
    fn __rust_alloc_zeroed(size: usize, align: usize) -> *mut u8;

    static __rust_no_alloc_shim_is_unstable: u8;
}

/// The global memory allocator.
///
/// This type implements the [`Allocator`] trait by forwarding calls
/// to the allocator registered with the `#[global_allocator]` attribute
/// if there is one, or the `std` crate’s default.
///
/// Note: while this type is unstable, the functionality it provides can be
/// accessed through the [free functions in `alloc`](self#functions).
#[unstable(feature = "allocator_api", issue = "32838")]
#[derive(Copy, Clone, Default, Debug)]
#[cfg(not(test))]
pub struct Global;

#[cfg(test)]
pub use std::alloc::Global;

/// Allocate memory with the global allocator.
///
/// This function forwards calls to the [`GlobalAlloc::alloc`] method
/// of the allocator registered with the `#[global_allocator]` attribute
/// if there is one, or the `std` crate’s default.
///
/// This function is expected to be deprecated in favor of the `alloc` method
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
#[stable(feature = "global_alloc", since = "1.28.0")]
#[must_use = "losing the pointer will leak memory"]
#[inline]
pub unsafe fn alloc(layout: Layout) -> *mut u8 {
    unsafe {
        // Make sure we don't accidentally allow omitting the allocator shim in
        // stable code until it is actually stabilized.
        core::ptr::read_volatile(&__rust_no_alloc_shim_is_unstable);

        __rust_alloc(layout.size(), layout.align())
    }
}

/// Deallocate memory with the global allocator.
///
/// This function forwards calls to the [`GlobalAlloc::dealloc`] method
/// of the allocator registered with the `#[global_allocator]` attribute
/// if there is one, or the `std` crate’s default.
///
/// This function is expected to be deprecated in favor of the `dealloc` method
/// of the [`Global`] type when it and the [`Allocator`] trait become stable.
///
/// # Safety
///
/// See [`GlobalAlloc::dealloc`].
#[stable(feature = "global_alloc", since = "1.28.0")]
#[inline]
pub unsafe fn dealloc(ptr: *mut u8, layout: Layout) {
    unsafe { __rust_dealloc(ptr, layout.size(), layout.align()) }
}

/// Reallocate memory with the global allocator.
///
/// This function forwards calls to the [`GlobalAlloc::realloc`] method
/// of the allocator registered with the `#[global_allocator]` attribute
/// if there is one, or the `std` crate’s default.
///
/// This function is expected to be deprecated in favor of the `realloc` method
/// of the [`Global`] type when it and the [`Allocator`] trait become stable.
///
/// # Safety
///
/// See [`GlobalAlloc::realloc`].
#[stable(feature = "global_alloc", since = "1.28.0")]
#[must_use = "losing the pointer will leak memory"]
#[inline]
pub unsafe fn realloc(ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
    unsafe { __rust_realloc(ptr, layout.size(), layout.align(), new_size) }
}

/// Allocate zero-initialized memory with the global allocator.
///
/// This function forwards calls to the [`GlobalAlloc::alloc_zeroed`] method
/// of the allocator registered with the `#[global_allocator]` attribute
/// if there is one, or the `std` crate’s default.
///
/// This function is expected to be deprecated in favor of the `alloc_zeroed` method
/// of the [`Global`] type when it and the [`Allocator`] trait become stable.
///
/// # Safety
///
/// See [`GlobalAlloc::alloc_zeroed`].
///
/// # Examples
///
/// ```
/// use std::alloc::{alloc_zeroed, dealloc, Layout};
///
/// unsafe {
///     let layout = Layout::new::<u16>();
///     let ptr = alloc_zeroed(layout);
///
///     assert_eq!(*(ptr as *mut u16), 0);
///
///     dealloc(ptr, layout);
/// }
/// ```
#[stable(feature = "global_alloc", since = "1.28.0")]
#[must_use = "losing the pointer will leak memory"]
#[inline]
pub unsafe fn alloc_zeroed(layout: Layout) -> *mut u8 {
    unsafe { __rust_alloc_zeroed(layout.size(), layout.align()) }
}

#[cfg(not(test))]
impl Global {
    #[inline]
    fn alloc_impl(&self, layout: Layout, zeroed: bool) -> Result<NonNull<[u8]>, AllocError> {
        match layout.size() {
            0 => Ok(NonNull::slice_from_raw_parts(layout.dangling(), 0)),
            // SAFETY: `layout` is non-zero in size,
            size => unsafe {
                let raw_ptr = if zeroed { alloc_zeroed(layout) } else { alloc(layout) };
                let ptr = NonNull::new(raw_ptr).ok_or(AllocError)?;
                Ok(NonNull::slice_from_raw_parts(ptr, size))
            },
        }
    }

    // SAFETY: Same as `Allocator::grow`
    #[inline]
    unsafe fn grow_impl(
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
                intrinsics::assume(new_size >= old_layout.size());

                let raw_ptr = realloc(ptr.as_ptr(), old_layout, new_size);
                let ptr = NonNull::new(raw_ptr).ok_or(AllocError)?;
                if zeroed {
                    raw_ptr.add(old_size).write_bytes(0, new_size - old_size);
                }
                Ok(NonNull::slice_from_raw_parts(ptr, new_size))
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
}

#[unstable(feature = "allocator_api", issue = "32838")]
#[cfg(not(test))]
unsafe impl Allocator for Global {
    #[inline]
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        self.alloc_impl(layout, false)
    }

    #[inline]
    fn allocate_zeroed(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        self.alloc_impl(layout, true)
    }

    #[inline]
    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        if layout.size() != 0 {
            // SAFETY: `layout` is non-zero in size,
            // other conditions must be upheld by the caller
            unsafe { dealloc(ptr.as_ptr(), layout) }
        }
    }

    #[inline]
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

        match new_layout.size() {
            // SAFETY: conditions must be upheld by the caller
            0 => unsafe {
                self.deallocate(ptr, old_layout);
                Ok(NonNull::slice_from_raw_parts(new_layout.dangling(), 0))
            },

            // SAFETY: `new_size` is non-zero. Other conditions must be upheld by the caller
            new_size if old_layout.align() == new_layout.align() => unsafe {
                // `realloc` probably checks for `new_size <= old_layout.size()` or something similar.
                intrinsics::assume(new_size <= old_layout.size());

                let raw_ptr = realloc(ptr.as_ptr(), old_layout, new_size);
                let ptr = NonNull::new(raw_ptr).ok_or(AllocError)?;
                Ok(NonNull::slice_from_raw_parts(ptr, new_size))
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

    #[cfg(not(no_global_oom_handling))]
    type ErrorHandling = Fatal;
    #[cfg(no_global_oom_handling)]
    type ErrorHandling = Fallible;
}

/// The allocator for unique pointers.
#[cfg(all(not(no_global_oom_handling), not(test)))]
#[lang = "exchange_malloc"]
#[inline]
unsafe fn exchange_malloc(size: usize, align: usize) -> *mut u8 {
    let layout = unsafe { Layout::from_size_align_unchecked(size, align) };
    match Global.allocate(layout) {
        Ok(ptr) => ptr.as_mut_ptr(),
        Err(_) => handle_alloc_error(layout),
    }
}

#[cfg_attr(not(test), lang = "box_free")]
#[inline]
// This signature has to be the same as `Box`, otherwise an ICE will happen.
// When an additional parameter to `Box` is added (like `A: Allocator`), this has to be added here as
// well.
// For example if `Box` is changed to  `struct Box<T: ?Sized, A: Allocator>(Unique<T>, A)`,
// this function has to be changed to `fn box_free<T: ?Sized, A: Allocator>(Unique<T>, A)` as well.
pub(crate) unsafe fn box_free<T: ?Sized, A: Allocator>(ptr: Unique<T>, alloc: A) {
    unsafe {
        let size = size_of_val(ptr.as_ref());
        let align = min_align_of_val(ptr.as_ref());
        let layout = Layout::from_size_align_unchecked(size, align);
        alloc.deallocate(From::from(ptr.cast()), layout)
    }
}

// # Allocation error handler

#[cfg(not(no_global_oom_handling))]
extern "Rust" {
    // This is the magic symbol to call the global alloc error handler. rustc generates
    // it to call `__rg_oom` if there is a `#[alloc_error_handler]`, or to call the
    // default implementations below (`__rdl_oom`) otherwise.
    fn __rust_alloc_error_handler(size: usize, align: usize) -> !;
}

/// Abort on memory allocation error or failure.
///
/// Callers of memory allocation APIs wishing to abort computation
/// in response to an allocation error are encouraged to call this function,
/// rather than directly invoking `panic!` or similar.
///
/// The default behavior of this function is to print a message to standard error
/// and abort the process.
/// It can be replaced with [`set_alloc_error_hook`] and [`take_alloc_error_hook`].
///
/// [`set_alloc_error_hook`]: ../../std/alloc/fn.set_alloc_error_hook.html
/// [`take_alloc_error_hook`]: ../../std/alloc/fn.take_alloc_error_hook.html
#[stable(feature = "global_alloc", since = "1.28.0")]
#[rustc_const_unstable(feature = "const_alloc_error", issue = "92523")]
#[cfg(all(not(no_global_oom_handling), not(test)))]
#[cold]
pub const fn handle_alloc_error(layout: Layout) -> ! {
    const fn ct_error(_: Layout) -> ! {
        panic!("allocation failed");
    }

    fn rt_error(layout: Layout) -> ! {
        unsafe {
            __rust_alloc_error_handler(layout.size(), layout.align());
        }
    }

    unsafe { core::intrinsics::const_eval_select((layout,), ct_error, rt_error) }
}

// For alloc test `std::alloc::handle_alloc_error` can be used directly.
#[cfg(all(not(no_global_oom_handling), test))]
pub use std::alloc::handle_alloc_error;

#[cfg(all(not(no_global_oom_handling), not(test)))]
#[doc(hidden)]
#[allow(unused_attributes)]
#[unstable(feature = "alloc_internals", issue = "none")]
pub mod __alloc_error_handler {
    // called via generated `__rust_alloc_error_handler` if there is no
    // `#[alloc_error_handler]`.
    #[rustc_std_internal_symbol]
    pub unsafe fn __rdl_oom(size: usize, _align: usize) -> ! {
        extern "Rust" {
            // This symbol is emitted by rustc next to __rust_alloc_error_handler.
            // Its value depends on the -Zoom={panic,abort} compiler option.
            static __rust_alloc_error_handler_should_panic: u8;
        }

        #[allow(unused_unsafe)]
        if unsafe { __rust_alloc_error_handler_should_panic != 0 } {
            panic!("memory allocation of {size} bytes failed")
        } else {
            core::panicking::panic_nounwind_fmt(format_args!(
                "memory allocation of {size} bytes failed"
            ))
        }
    }
}

/// Specialize clones into pre-allocated, uninitialized memory.
/// Used by `Box::clone` and `Rc`/`Arc::make_mut`.
pub(crate) trait WriteCloneIntoRaw: Sized {
    unsafe fn write_clone_into_raw(&self, target: *mut Self);
}

impl<T: Clone> WriteCloneIntoRaw for T {
    #[inline]
    default unsafe fn write_clone_into_raw(&self, target: *mut Self) {
        // Having allocated *first* may allow the optimizer to create
        // the cloned value in-place, skipping the local and move.
        unsafe { target.write(self.clone()) };
    }
}

impl<T: Copy> WriteCloneIntoRaw for T {
    #[inline]
    unsafe fn write_clone_into_raw(&self, target: *mut Self) {
        // We can always copy in-place, without ever involving a local value.
        unsafe { target.copy_from_nonoverlapping(self, 1) };
    }
}

#[cfg(all(not(no_global_oom_handling), not(test)))]
use core::error::Error;

/// Trait for handling alloc errors for allocators which
/// panic or abort instead of returning errors.
#[unstable(feature = "allocator_api", issue = "32838")]
#[cfg(all(not(no_global_oom_handling), not(test)))]
#[rustc_specialization_trait]
pub trait HandleAllocError: Error {
    /// Globally handle this allocation error
    fn handle_alloc_error(self) -> !;
}

/// Error handling mode to use when the user of the type wants to ignore
/// allocation failures, treating them as a fatal error. Functions
/// performing allocation will return values directly.
#[derive(Debug)]
#[unstable(feature = "allocator_api", issue = "32838")]
#[cfg(all(not(no_global_oom_handling), not(test)))]
pub struct Fatal;

#[unstable(feature = "alloc_internals", issue = "none")]
#[cfg(all(not(no_global_oom_handling), not(test)))]
impl error_handling_sealed::Sealed for Fatal {}

#[unstable(feature = "allocator_api", issue = "32838")]
#[cfg(all(not(no_global_oom_handling), not(test)))]
impl ErrorHandling for Fatal {
    type Result<T, E: Error> = T;

    fn map_result<T, E: Error>(result: Result<T, E>) -> Self::Result<T, E> {
        /// Hack around lack of `cfg(no_global_oom_handling)` in core.
        ///
        /// Using post-monomorphization errors and specialization,
        /// we can enforce that any error used with `Fatal` implements
        /// `HandleAllocError`, without requiring that all errors used
        /// with fallible allocation implement it. This also allows
        /// for `HandleAllocError` to live with the rest of the
        /// global allocation handling in the `alloc` crate.
        trait HandleAllocErrorInternal {
            fn handle_alloc_error_internal(self) -> !;
        }
        impl<E: Error> HandleAllocErrorInternal for E {
            default fn handle_alloc_error_internal(self) -> ! {
                const {
                    panic!(
                        "user must implement `HandleAllocError` for any error type used with the `Fatal` kind of `ErrorHandling`"
                    )
                }
            }
        }
        impl<E: HandleAllocError> HandleAllocErrorInternal for E {
            fn handle_alloc_error_internal(self) -> ! {
                self.handle_alloc_error()
            }
        }

        result.unwrap_or_else(|e| e.handle_alloc_error_internal())
    }
}

/// Wrapper around an existing allocator allowing one to
/// use a fallible allocator as an infallible one.
#[unstable(feature = "allocator_api", issue = "32838")]
#[cfg(all(not(no_global_oom_handling), not(test)))]
#[derive(Debug)]
pub struct FatalAdapter<A: Allocator<ErrorHandling = Fallible>>(pub A);

#[unstable(feature = "allocator_api", issue = "32838")]
#[cfg(all(not(no_global_oom_handling), not(test)))]
unsafe impl<A: Allocator<ErrorHandling = Fallible>> Allocator for FatalAdapter<A> {
    fn allocate(&self, layout: Layout) -> Result<core::ptr::NonNull<[u8]>, AllocError> {
        self.0.allocate(layout)
    }

    fn allocate_zeroed(&self, layout: Layout) -> Result<core::ptr::NonNull<[u8]>, AllocError> {
        self.0.allocate_zeroed(layout)
    }

    unsafe fn deallocate(&self, ptr: core::ptr::NonNull<u8>, layout: Layout) {
        // SAFETY: the safety contract must be upheld by the caller
        unsafe { self.0.deallocate(ptr, layout) }
    }

    unsafe fn grow(
        &self,
        ptr: core::ptr::NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<core::ptr::NonNull<[u8]>, AllocError> {
        // SAFETY: the safety contract must be upheld by the caller
        unsafe { self.0.grow(ptr, old_layout, new_layout) }
    }

    unsafe fn grow_zeroed(
        &self,
        ptr: core::ptr::NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<core::ptr::NonNull<[u8]>, AllocError> {
        // SAFETY: the safety contract must be upheld by the caller
        unsafe { self.0.grow_zeroed(ptr, old_layout, new_layout) }
    }

    unsafe fn shrink(
        &self,
        ptr: core::ptr::NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<core::ptr::NonNull<[u8]>, AllocError> {
        // SAFETY: the safety contract must be upheld by the caller
        unsafe { self.0.shrink(ptr, old_layout, new_layout) }
    }

    fn by_ref(&self) -> &Self
    where
        Self: Sized,
    {
        self
    }

    type ErrorHandling = Fatal;
}

/// Wrapper around an existing allocator allowing one to
/// use an infallible allocator as a fallible one.
#[unstable(feature = "allocator_api", issue = "32838")]
#[cfg(all(not(no_global_oom_handling), not(test)))]
#[derive(Debug)]
pub struct FallibleAdapter<A: Allocator<ErrorHandling = Fatal>>(pub A);

#[unstable(feature = "allocator_api", issue = "32838")]
#[cfg(all(not(no_global_oom_handling), not(test)))]
unsafe impl<A: Allocator<ErrorHandling = Fatal>> Allocator for FallibleAdapter<A> {
    fn allocate(&self, layout: Layout) -> Result<core::ptr::NonNull<[u8]>, AllocError> {
        self.0.allocate(layout)
    }

    fn allocate_zeroed(&self, layout: Layout) -> Result<core::ptr::NonNull<[u8]>, AllocError> {
        self.0.allocate_zeroed(layout)
    }

    unsafe fn deallocate(&self, ptr: core::ptr::NonNull<u8>, layout: Layout) {
        // SAFETY: the safety contract must be upheld by the caller
        unsafe { self.0.deallocate(ptr, layout) }
    }

    unsafe fn grow(
        &self,
        ptr: core::ptr::NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<core::ptr::NonNull<[u8]>, AllocError> {
        // SAFETY: the safety contract must be upheld by the caller
        unsafe { self.0.grow(ptr, old_layout, new_layout) }
    }

    unsafe fn grow_zeroed(
        &self,
        ptr: core::ptr::NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<core::ptr::NonNull<[u8]>, AllocError> {
        // SAFETY: the safety contract must be upheld by the caller
        unsafe { self.0.grow_zeroed(ptr, old_layout, new_layout) }
    }

    unsafe fn shrink(
        &self,
        ptr: core::ptr::NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<core::ptr::NonNull<[u8]>, AllocError> {
        // SAFETY: the safety contract must be upheld by the caller
        unsafe { self.0.shrink(ptr, old_layout, new_layout) }
    }

    fn by_ref(&self) -> &Self
    where
        Self: Sized,
    {
        self
    }

    type ErrorHandling = Fallible;
}

#[cfg(test)]
pub use std::alloc::{FallibleAdapter, Fatal, FatalAdapter, HandleAllocError};

#[cfg(not(no_global_oom_handling))]
use crate::collections::{TryReserveError, TryReserveErrorKind};

// One central function responsible for reporting capacity overflows. This'll
// ensure that the code generation related to these panics is minimal as there's
// only one location which panics rather than a bunch throughout the module.
#[cfg(not(no_global_oom_handling))]
pub(crate) fn capacity_overflow() -> ! {
    panic!("capacity overflow");
}

#[cfg(not(no_global_oom_handling))]
#[unstable(feature = "allocator_api", issue = "32838")]
impl HandleAllocError for TryReserveError {
    fn handle_alloc_error(self) -> ! {
        match self.kind() {
            TryReserveErrorKind::CapacityOverflow => capacity_overflow(),
            TryReserveErrorKind::AllocError { layout, .. } => handle_alloc_error(layout),
        }
    }
}

pub(crate) type AllocResult<A, T, E> =
    <<A as Allocator>::ErrorHandling as ErrorHandling>::Result<T, E>;
