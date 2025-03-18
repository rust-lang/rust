//! Memory allocation APIs.
//!
//! In a given program, the standard library has one “global” memory allocator
//! that is used for example by `Box<T>` and `Vec<T>`.
//!
//! Currently the default global allocator is unspecified. Libraries, however,
//! like `cdylib`s and `staticlib`s are guaranteed to use the [`System`] by
//! default.
//!
//! # The `#[global_allocator]` attribute
//!
//! This attribute allows configuring the choice of global allocator.
//! You can use this to implement a completely custom global allocator
//! to route all default allocation requests to a custom object.
//!
//! ```rust
//! use std::alloc::{GlobalAlloc, System, Layout};
//!
//! struct MyAllocator;
//!
//! unsafe impl GlobalAlloc for MyAllocator {
//!     unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
//!         unsafe { System.alloc(layout) }
//!     }
//!
//!     unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
//!         unsafe { System.dealloc(ptr, layout) }
//!     }
//! }
//!
//! #[global_allocator]
//! static GLOBAL: MyAllocator = MyAllocator;
//!
//! fn main() {
//!     // This `Vec` will allocate memory through `GLOBAL` above
//!     let mut v = Vec::new();
//!     v.push(1);
//! }
//! ```
//!
//! The attribute is used on a `static` item whose type implements the
//! [`GlobalAlloc`] trait. This type can be provided by an external library:
//!
//! ```rust,ignore (demonstrates crates.io usage)
//! use jemallocator::Jemalloc;
//!
//! #[global_allocator]
//! static GLOBAL: Jemalloc = Jemalloc;
//!
//! fn main() {}
//! ```
//!
//! The `#[global_allocator]` can only be used once in a crate
//! or its recursive dependencies.

#![deny(unsafe_op_in_unsafe_fn)]
#![stable(feature = "alloc_module", since = "1.28.0")]

use core::ptr::NonNull;
use core::sync::atomic::{AtomicPtr, Ordering};
use core::{hint, mem, ptr};

#[stable(feature = "alloc_module", since = "1.28.0")]
#[doc(inline)]
pub use alloc_crate::alloc::*;

/// The default memory allocator provided by the operating system.
///
/// This is based on `malloc` on Unix platforms and `HeapAlloc` on Windows,
/// plus related functions. However, it is not valid to mix use of the backing
/// system allocator with `System`, as this implementation may include extra
/// work, such as to serve alignment requests greater than the alignment
/// provided directly by the backing system allocator.
///
/// This type implements the [`GlobalAlloc`] trait. Currently the default
/// global allocator is unspecified. Libraries, however, like `cdylib`s and
/// `staticlib`s are guaranteed to use the [`System`] by default and as such
/// work as if they had this definition:
///
/// ```rust
/// use std::alloc::System;
///
/// #[global_allocator]
/// static A: System = System;
///
/// fn main() {
///     let a = Box::new(4); // Allocates from the system allocator.
///     println!("{a}");
/// }
/// ```
///
/// You can also define your own wrapper around `System` if you'd like, such as
/// keeping track of the number of all bytes allocated:
///
/// ```rust
/// use std::alloc::{System, GlobalAlloc, Layout};
/// use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};
///
/// struct Counter;
///
/// static ALLOCATED: AtomicUsize = AtomicUsize::new(0);
///
/// unsafe impl GlobalAlloc for Counter {
///     unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
///         let ret = unsafe { System.alloc(layout) };
///         if !ret.is_null() {
///             ALLOCATED.fetch_add(layout.size(), Relaxed);
///         }
///         ret
///     }
///
///     unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
///         unsafe { System.dealloc(ptr, layout); }
///         ALLOCATED.fetch_sub(layout.size(), Relaxed);
///     }
/// }
///
/// #[global_allocator]
/// static A: Counter = Counter;
///
/// fn main() {
///     println!("allocated bytes before main: {}", ALLOCATED.load(Relaxed));
/// }
/// ```
///
/// It can also be used directly to allocate memory independently of whatever
/// global allocator has been selected for a Rust program. For example if a Rust
/// program opts in to using jemalloc as the global allocator, `System` will
/// still allocate memory using `malloc` and `HeapAlloc`.
#[stable(feature = "alloc_system_type", since = "1.28.0")]
#[derive(Debug, Default, Copy, Clone)]
pub struct System;

impl System {
    #[inline]
    fn alloc_impl(&self, layout: Layout, zeroed: bool) -> Result<NonNull<[u8]>, AllocError> {
        match layout.size() {
            0 => Ok(NonNull::slice_from_raw_parts(layout.dangling(), 0)),
            // SAFETY: `layout` is non-zero in size,
            size => unsafe {
                let raw_ptr = if zeroed {
                    GlobalAlloc::alloc_zeroed(self, layout)
                } else {
                    GlobalAlloc::alloc(self, layout)
                };
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

            // SAFETY: `new_size` is non-zero as `new_size` is greater than or equal to `old_size`
            // as required by safety conditions and the `old_size == 0` case was handled in the
            // previous match arm. Other conditions must be upheld by the caller
            old_size if old_layout.align() == new_layout.align() => unsafe {
                let new_size = new_layout.size();

                // `realloc` probably checks for `new_size >= old_layout.size()` or something similar.
                hint::assert_unchecked(new_size >= old_layout.size());

                let raw_ptr = GlobalAlloc::realloc(self, ptr.as_ptr(), old_layout, new_size);
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
                Allocator::deallocate(self, ptr, old_layout);
                Ok(new_ptr)
            },
        }
    }
}

// The Allocator impl checks the layout size to be non-zero and forwards to the GlobalAlloc impl,
// which is in `std::sys::*::alloc`.
#[unstable(feature = "allocator_api", issue = "32838")]
unsafe impl Allocator for System {
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
            unsafe { GlobalAlloc::dealloc(self, ptr.as_ptr(), layout) }
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
                Allocator::deallocate(self, ptr, old_layout);
                Ok(NonNull::slice_from_raw_parts(new_layout.dangling(), 0))
            },

            // SAFETY: `new_size` is non-zero. Other conditions must be upheld by the caller
            new_size if old_layout.align() == new_layout.align() => unsafe {
                // `realloc` probably checks for `new_size <= old_layout.size()` or something similar.
                hint::assert_unchecked(new_size <= old_layout.size());

                let raw_ptr = GlobalAlloc::realloc(self, ptr.as_ptr(), old_layout, new_size);
                let ptr = NonNull::new(raw_ptr).ok_or(AllocError)?;
                Ok(NonNull::slice_from_raw_parts(ptr, new_size))
            },

            // SAFETY: because `new_size` must be smaller than or equal to `old_layout.size()`,
            // both the old and new memory allocation are valid for reads and writes for `new_size`
            // bytes. Also, because the old allocation wasn't yet deallocated, it cannot overlap
            // `new_ptr`. Thus, the call to `copy_nonoverlapping` is safe. The safety contract
            // for `dealloc` must be upheld by the caller.
            new_size => unsafe {
                let new_ptr = Allocator::allocate(self, new_layout)?;
                ptr::copy_nonoverlapping(ptr.as_ptr(), new_ptr.as_mut_ptr(), new_size);
                Allocator::deallocate(self, ptr, old_layout);
                Ok(new_ptr)
            },
        }
    }
}

static HOOK: AtomicPtr<()> = AtomicPtr::new(ptr::null_mut());

/// Registers a custom allocation error hook, replacing any that was previously registered.
///
/// The allocation error hook is invoked when an infallible memory allocation fails — that is,
/// as a consequence of calling [`handle_alloc_error`] — before the runtime aborts.
///
/// The allocation error hook is a global resource. [`take_alloc_error_hook`] may be used to
/// retrieve a previously registered hook and wrap or discard it.
///
/// # What the provided `hook` function should expect
///
/// The hook function is provided with a [`Layout`] struct which contains information
/// about the allocation that failed.
///
/// The hook function may choose to panic or abort; in the event that it returns normally, this
/// will cause an immediate abort.
///
/// Since [`take_alloc_error_hook`] is a safe function that allows retrieving the hook, the hook
/// function must be _sound_ to call even if no memory allocations were attempted.
///
/// # The default hook
///
/// The default hook, used if [`set_alloc_error_hook`] is never called, prints a message to
/// standard error (and then returns, causing the runtime to abort the process).
/// Compiler options may cause it to panic instead, and the default behavior may be changed
/// to panicking in future versions of Rust.
///
/// # Examples
///
/// ```
/// #![feature(alloc_error_hook)]
///
/// use std::alloc::{Layout, set_alloc_error_hook};
///
/// fn custom_alloc_error_hook(layout: Layout) {
///    panic!("memory allocation of {} bytes failed", layout.size());
/// }
///
/// set_alloc_error_hook(custom_alloc_error_hook);
/// ```
#[unstable(feature = "alloc_error_hook", issue = "51245")]
pub fn set_alloc_error_hook(hook: fn(Layout)) {
    HOOK.store(hook as *mut (), Ordering::Release);
}

/// Unregisters the current allocation error hook, returning it.
///
/// *See also the function [`set_alloc_error_hook`].*
///
/// If no custom hook is registered, the default hook will be returned.
#[unstable(feature = "alloc_error_hook", issue = "51245")]
pub fn take_alloc_error_hook() -> fn(Layout) {
    let hook = HOOK.swap(ptr::null_mut(), Ordering::Acquire);
    if hook.is_null() { default_alloc_error_hook } else { unsafe { mem::transmute(hook) } }
}

fn default_alloc_error_hook(layout: Layout) {
    unsafe extern "Rust" {
        // This symbol is emitted by rustc next to __rust_alloc_error_handler.
        // Its value depends on the -Zoom={panic,abort} compiler option.
        #[cfg_attr(not(bootstrap), rustc_std_internal_symbol)]
        static __rust_alloc_error_handler_should_panic: u8;
    }

    if unsafe { __rust_alloc_error_handler_should_panic != 0 } {
        panic!("memory allocation of {} bytes failed", layout.size());
    } else {
        // This is the default path taken on OOM, and the only path taken on stable with std.
        // Crucially, it does *not* call any user-defined code, and therefore users do not have to
        // worry about allocation failure causing reentrancy issues. That makes it different from
        // the default `__rdl_oom` defined in alloc (i.e., the default alloc error handler that is
        // called when there is no `#[alloc_error_handler]`), which triggers a regular panic and
        // thus can invoke a user-defined panic hook, executing arbitrary user-defined code.
        rtprintpanic!("memory allocation of {} bytes failed\n", layout.size());
    }
}

#[cfg(not(test))]
#[doc(hidden)]
#[alloc_error_handler]
#[unstable(feature = "alloc_internals", issue = "none")]
pub fn rust_oom(layout: Layout) -> ! {
    let hook = HOOK.load(Ordering::Acquire);
    let hook: fn(Layout) =
        if hook.is_null() { default_alloc_error_hook } else { unsafe { mem::transmute(hook) } };
    hook(layout);
    crate::process::abort()
}

#[cfg(not(test))]
#[doc(hidden)]
#[allow(unused_attributes)]
#[unstable(feature = "alloc_internals", issue = "none")]
pub mod __default_lib_allocator {
    use super::{GlobalAlloc, Layout, System};
    // These magic symbol names are used as a fallback for implementing the
    // `__rust_alloc` etc symbols (see `src/liballoc/alloc.rs`) when there is
    // no `#[global_allocator]` attribute.

    // for symbol names src/librustc_ast/expand/allocator.rs
    // for signatures src/librustc_allocator/lib.rs

    // linkage directives are provided as part of the current compiler allocator
    // ABI

    #[rustc_std_internal_symbol]
    pub unsafe extern "C" fn __rdl_alloc(size: usize, align: usize) -> *mut u8 {
        // SAFETY: see the guarantees expected by `Layout::from_size_align` and
        // `GlobalAlloc::alloc`.
        unsafe {
            let layout = Layout::from_size_align_unchecked(size, align);
            System.alloc(layout)
        }
    }

    #[rustc_std_internal_symbol]
    pub unsafe extern "C" fn __rdl_dealloc(ptr: *mut u8, size: usize, align: usize) {
        // SAFETY: see the guarantees expected by `Layout::from_size_align` and
        // `GlobalAlloc::dealloc`.
        unsafe { System.dealloc(ptr, Layout::from_size_align_unchecked(size, align)) }
    }

    #[rustc_std_internal_symbol]
    pub unsafe extern "C" fn __rdl_realloc(
        ptr: *mut u8,
        old_size: usize,
        align: usize,
        new_size: usize,
    ) -> *mut u8 {
        // SAFETY: see the guarantees expected by `Layout::from_size_align` and
        // `GlobalAlloc::realloc`.
        unsafe {
            let old_layout = Layout::from_size_align_unchecked(old_size, align);
            System.realloc(ptr, old_layout, new_size)
        }
    }

    #[rustc_std_internal_symbol]
    pub unsafe extern "C" fn __rdl_alloc_zeroed(size: usize, align: usize) -> *mut u8 {
        // SAFETY: see the guarantees expected by `Layout::from_size_align` and
        // `GlobalAlloc::alloc_zeroed`.
        unsafe {
            let layout = Layout::from_size_align_unchecked(size, align);
            System.alloc_zeroed(layout)
        }
    }
}
