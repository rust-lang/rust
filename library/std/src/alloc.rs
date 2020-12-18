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
//!         System.alloc(layout)
//!     }
//!
//!     unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
//!         System.dealloc(ptr, layout)
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
//! extern crate jemallocator;
//!
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

use core::intrinsics;
use core::ptr::NonNull;
use core::sync::atomic::{AtomicPtr, Ordering};
use core::{mem, ptr};

use crate::sys_common::util::dumb_print;

#[stable(feature = "alloc_module", since = "1.28.0")]
#[doc(inline)]
pub use alloc_crate::alloc::*;

/// The default memory allocator provided by the operating system.
///
/// This is based on `malloc` on Unix platforms and `HeapAlloc` on Windows,
/// plus related functions.
///
/// This type implements the `GlobalAlloc` trait and Rust programs by default
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
///     println!("{}", a);
/// }
/// ```
///
/// You can also define your own wrapper around `System` if you'd like, such as
/// keeping track of the number of all bytes allocated:
///
/// ```rust
/// use std::alloc::{System, GlobalAlloc, Layout};
/// use std::sync::atomic::{AtomicUsize, Ordering::SeqCst};
///
/// struct Counter;
///
/// static ALLOCATED: AtomicUsize = AtomicUsize::new(0);
///
/// unsafe impl GlobalAlloc for Counter {
///     unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
///         let ret = System.alloc(layout);
///         if !ret.is_null() {
///             ALLOCATED.fetch_add(layout.size(), SeqCst);
///         }
///         return ret
///     }
///
///     unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
///         System.dealloc(ptr, layout);
///         ALLOCATED.fetch_sub(layout.size(), SeqCst);
///     }
/// }
///
/// #[global_allocator]
/// static A: Counter = Counter;
///
/// fn main() {
///     println!("allocated bytes before main: {}", ALLOCATED.load(SeqCst));
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

            // SAFETY: `new_size` is non-zero as `old_size` is greater than or equal to `new_size`
            // as required by safety conditions. Other conditions must be upheld by the caller
            old_size if old_layout.align() == new_layout.align() => unsafe {
                let new_size = new_layout.size();

                // `realloc` probably checks for `new_size >= old_layout.size()` or something similar.
                intrinsics::assume(new_size >= old_layout.size());

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
                Allocator::deallocate(&self, ptr, old_layout);
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
                Allocator::deallocate(&self, ptr, old_layout);
                Ok(NonNull::slice_from_raw_parts(new_layout.dangling(), 0))
            },

            // SAFETY: `new_size` is non-zero. Other conditions must be upheld by the caller
            new_size if old_layout.align() == new_layout.align() => unsafe {
                // `realloc` probably checks for `new_size <= old_layout.size()` or something similar.
                intrinsics::assume(new_size <= old_layout.size());

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
                let new_ptr = Allocator::allocate(&self, new_layout)?;
                ptr::copy_nonoverlapping(ptr.as_ptr(), new_ptr.as_mut_ptr(), new_size);
                Allocator::deallocate(&self, ptr, old_layout);
                Ok(new_ptr)
            },
        }
    }
}

static HOOK: AtomicPtr<()> = AtomicPtr::new(ptr::null_mut());

/// Registers a custom allocation error hook, replacing any that was previously registered.
///
/// The allocation error hook is invoked when an infallible memory allocation fails, before
/// the runtime aborts. The default hook prints a message to standard error,
/// but this behavior can be customized with the [`set_alloc_error_hook`] and
/// [`take_alloc_error_hook`] functions.
///
/// The hook is provided with a `Layout` struct which contains information
/// about the allocation that failed.
///
/// The allocation error hook is a global resource.
#[unstable(feature = "alloc_error_hook", issue = "51245")]
pub fn set_alloc_error_hook(hook: fn(Layout)) {
    HOOK.store(hook as *mut (), Ordering::SeqCst);
}

/// Unregisters the current allocation error hook, returning it.
///
/// *See also the function [`set_alloc_error_hook`].*
///
/// If no custom hook is registered, the default hook will be returned.
#[unstable(feature = "alloc_error_hook", issue = "51245")]
pub fn take_alloc_error_hook() -> fn(Layout) {
    let hook = HOOK.swap(ptr::null_mut(), Ordering::SeqCst);
    if hook.is_null() { default_alloc_error_hook } else { unsafe { mem::transmute(hook) } }
}

fn default_alloc_error_hook(layout: Layout) {
    dumb_print(format_args!("memory allocation of {} bytes failed\n", layout.size()));
}

#[cfg(not(test))]
#[doc(hidden)]
#[alloc_error_handler]
#[unstable(feature = "alloc_internals", issue = "none")]
pub fn rust_oom(layout: Layout) -> ! {
    let hook = HOOK.load(Ordering::SeqCst);
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
