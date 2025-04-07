//! Memory allocation APIs

#![stable(feature = "alloc_module", since = "1.28.0")]

#[stable(feature = "alloc_module", since = "1.28.0")]
#[doc(inline)]
pub use core::alloc::*;
use core::hint;
use core::ptr::{self, NonNull};

unsafe extern "Rust" {
    // These are the magic symbols to call the global allocator. rustc generates
    // them to call the global allocator if there is a `#[global_allocator]` attribute
    // (the code expanding that attribute macro generates those functions), or to call
    // the default implementations in std (`__rdl_alloc` etc. in `library/std/src/alloc.rs`)
    // otherwise.
    #[rustc_allocator]
    #[rustc_nounwind]
    #[cfg_attr(not(bootstrap), rustc_std_internal_symbol)]
    fn __rust_alloc(size: usize, align: usize) -> *mut u8;
    #[rustc_deallocator]
    #[rustc_nounwind]
    #[cfg_attr(not(bootstrap), rustc_std_internal_symbol)]
    fn __rust_dealloc(ptr: *mut u8, size: usize, align: usize);
    #[rustc_reallocator]
    #[rustc_nounwind]
    #[cfg_attr(not(bootstrap), rustc_std_internal_symbol)]
    fn __rust_realloc(ptr: *mut u8, old_size: usize, align: usize, new_size: usize) -> *mut u8;
    #[rustc_allocator_zeroed]
    #[rustc_nounwind]
    #[cfg_attr(not(bootstrap), rustc_std_internal_symbol)]
    fn __rust_alloc_zeroed(size: usize, align: usize) -> *mut u8;

    #[cfg_attr(not(bootstrap), rustc_std_internal_symbol)]
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
// the compiler needs to know when a Box uses the global allocator vs a custom one
#[lang = "global_alloc_ty"]
pub struct Global;

/// Allocates memory with the global allocator.
///
/// This function forwards calls to the [`GlobalAlloc::alloc`] method
/// of the allocator registered with the `#[global_allocator]` attribute
/// if there is one, or the `std` crate’s default.
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
#[stable(feature = "global_alloc", since = "1.28.0")]
#[must_use = "losing the pointer will leak memory"]
#[inline]
#[cfg_attr(miri, track_caller)] // even without panics, this helps for Miri backtraces
pub unsafe fn alloc(layout: Layout) -> *mut u8 {
    unsafe {
        // Make sure we don't accidentally allow omitting the allocator shim in
        // stable code until it is actually stabilized.
        core::ptr::read_volatile(&__rust_no_alloc_shim_is_unstable);

        __rust_alloc(layout.size(), layout.align())
    }
}

/// Deallocates memory with the global allocator.
///
/// This function forwards calls to the [`GlobalAlloc::dealloc`] method
/// of the allocator registered with the `#[global_allocator]` attribute
/// if there is one, or the `std` crate’s default.
///
/// This function is expected to be deprecated in favor of the `deallocate` method
/// of the [`Global`] type when it and the [`Allocator`] trait become stable.
///
/// # Safety
///
/// See [`GlobalAlloc::dealloc`].
#[stable(feature = "global_alloc", since = "1.28.0")]
#[inline]
#[cfg_attr(miri, track_caller)] // even without panics, this helps for Miri backtraces
pub unsafe fn dealloc(ptr: *mut u8, layout: Layout) {
    unsafe { __rust_dealloc(ptr, layout.size(), layout.align()) }
}

/// Reallocates memory with the global allocator.
///
/// This function forwards calls to the [`GlobalAlloc::realloc`] method
/// of the allocator registered with the `#[global_allocator]` attribute
/// if there is one, or the `std` crate’s default.
///
/// This function is expected to be deprecated in favor of the `grow` and `shrink` methods
/// of the [`Global`] type when it and the [`Allocator`] trait become stable.
///
/// # Safety
///
/// See [`GlobalAlloc::realloc`].
#[stable(feature = "global_alloc", since = "1.28.0")]
#[must_use = "losing the pointer will leak memory"]
#[inline]
#[cfg_attr(miri, track_caller)] // even without panics, this helps for Miri backtraces
pub unsafe fn realloc(ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
    unsafe { __rust_realloc(ptr, layout.size(), layout.align(), new_size) }
}

/// Allocates zero-initialized memory with the global allocator.
///
/// This function forwards calls to the [`GlobalAlloc::alloc_zeroed`] method
/// of the allocator registered with the `#[global_allocator]` attribute
/// if there is one, or the `std` crate’s default.
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
#[stable(feature = "global_alloc", since = "1.28.0")]
#[must_use = "losing the pointer will leak memory"]
#[inline]
#[cfg_attr(miri, track_caller)] // even without panics, this helps for Miri backtraces
pub unsafe fn alloc_zeroed(layout: Layout) -> *mut u8 {
    unsafe {
        // Make sure we don't accidentally allow omitting the allocator shim in
        // stable code until it is actually stabilized.
        core::ptr::read_volatile(&__rust_no_alloc_shim_is_unstable);

        __rust_alloc_zeroed(layout.size(), layout.align())
    }
}

impl Global {
    #[inline]
    #[cfg_attr(miri, track_caller)] // even without panics, this helps for Miri backtraces
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
    #[cfg_attr(miri, track_caller)] // even without panics, this helps for Miri backtraces
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
                hint::assert_unchecked(new_size >= old_layout.size());

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
unsafe impl Allocator for Global {
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
        if layout.size() != 0 {
            // SAFETY:
            // * We have checked that `layout` is non-zero in size.
            // * The caller is obligated to provide a layout that "fits", and in this case,
            //   "fit" always means a layout that is equal to the original, because our
            //   `allocate()`, `grow()`, and `shrink()` implementations never returns a larger
            //   allocation than requested.
            // * Other conditions must be upheld by the caller, as per `Allocator::deallocate()`'s
            //   safety documentation.
            unsafe { dealloc(ptr.as_ptr(), layout) }
        }
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
                hint::assert_unchecked(new_size <= old_layout.size());

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
}

/// The allocator for `Box`.
#[cfg(not(no_global_oom_handling))]
#[lang = "exchange_malloc"]
#[inline]
#[cfg_attr(miri, track_caller)] // even without panics, this helps for Miri backtraces
unsafe fn exchange_malloc(size: usize, align: usize) -> *mut u8 {
    let layout = unsafe { Layout::from_size_align_unchecked(size, align) };
    match Global.allocate(layout) {
        Ok(ptr) => ptr.as_mut_ptr(),
        Err(_) => handle_alloc_error(layout),
    }
}

// # Allocation error handler

#[cfg(not(no_global_oom_handling))]
unsafe extern "Rust" {
    // This is the magic symbol to call the global alloc error handler. rustc generates
    // it to call `__rg_oom` if there is a `#[alloc_error_handler]`, or to call the
    // default implementations below (`__rdl_oom`) otherwise.
    #[cfg_attr(not(bootstrap), rustc_std_internal_symbol)]
    fn __rust_alloc_error_handler(size: usize, align: usize) -> !;
}

/// Signals a memory allocation error.
///
/// Callers of memory allocation APIs wishing to cease execution
/// in response to an allocation error are encouraged to call this function,
/// rather than directly invoking [`panic!`] or similar.
///
/// This function is guaranteed to diverge (not return normally with a value), but depending on
/// global configuration, it may either panic (resulting in unwinding or aborting as per
/// configuration for all panics), or abort the process (with no unwinding).
///
/// The default behavior is:
///
///  * If the binary links against `std` (typically the case), then
///   print a message to standard error and abort the process.
///   This behavior can be replaced with [`set_alloc_error_hook`] and [`take_alloc_error_hook`].
///   Future versions of Rust may panic by default instead.
///
/// * If the binary does not link against `std` (all of its crates are marked
///   [`#![no_std]`][no_std]), then call [`panic!`] with a message.
///   [The panic handler] applies as to any panic.
///
/// [`set_alloc_error_hook`]: ../../std/alloc/fn.set_alloc_error_hook.html
/// [`take_alloc_error_hook`]: ../../std/alloc/fn.take_alloc_error_hook.html
/// [The panic handler]: https://doc.rust-lang.org/reference/runtime.html#the-panic_handler-attribute
/// [no_std]: https://doc.rust-lang.org/reference/names/preludes.html#the-no_std-attribute
#[stable(feature = "global_alloc", since = "1.28.0")]
#[rustc_const_unstable(feature = "const_alloc_error", issue = "92523")]
#[cfg(not(no_global_oom_handling))]
#[cold]
#[optimize(size)]
pub const fn handle_alloc_error(layout: Layout) -> ! {
    const fn ct_error(_: Layout) -> ! {
        panic!("allocation failed");
    }

    #[inline]
    fn rt_error(layout: Layout) -> ! {
        unsafe {
            __rust_alloc_error_handler(layout.size(), layout.align());
        }
    }

    #[cfg(not(feature = "panic_immediate_abort"))]
    {
        core::intrinsics::const_eval_select((layout,), ct_error, rt_error)
    }

    #[cfg(feature = "panic_immediate_abort")]
    ct_error(layout)
}

#[cfg(not(no_global_oom_handling))]
#[doc(hidden)]
#[allow(unused_attributes)]
#[unstable(feature = "alloc_internals", issue = "none")]
pub mod __alloc_error_handler {
    // called via generated `__rust_alloc_error_handler` if there is no
    // `#[alloc_error_handler]`.
    #[rustc_std_internal_symbol]
    pub unsafe fn __rdl_oom(size: usize, _align: usize) -> ! {
        unsafe extern "Rust" {
            // This symbol is emitted by rustc next to __rust_alloc_error_handler.
            // Its value depends on the -Zoom={panic,abort} compiler option.
            #[cfg_attr(not(bootstrap), rustc_std_internal_symbol)]
            static __rust_alloc_error_handler_should_panic: u8;
        }

        if unsafe { __rust_alloc_error_handler_should_panic != 0 } {
            panic!("memory allocation of {size} bytes failed")
        } else {
            core::panicking::panic_nounwind_fmt(
                format_args!("memory allocation of {size} bytes failed"),
                /* force_no_backtrace */ false,
            )
        }
    }
}
