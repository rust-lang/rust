//! Memory allocation APIs

#![stable(feature = "alloc_module", since = "1.28.0")]

use core::intrinsics::{self, min_align_of_val, size_of_val};
use core::ptr::{NonNull, Unique};

#[stable(feature = "alloc_module", since = "1.28.0")]
#[doc(inline)]
pub use core::alloc::*;

#[cfg(test)]
mod tests;

extern "Rust" {
    // These are the magic symbols to call the global allocator.  rustc generates
    // them from the `#[global_allocator]` attribute if there is one, or uses the
    // default implementations in libstd (`__rdl_alloc` etc in `src/libstd/alloc.rs`)
    // otherwise.
    #[rustc_allocator]
    #[rustc_allocator_nounwind]
    fn __rust_alloc(size: usize, align: usize) -> *mut u8;
    #[rustc_allocator_nounwind]
    fn __rust_dealloc(ptr: *mut u8, size: usize, align: usize);
    #[rustc_allocator_nounwind]
    fn __rust_realloc(ptr: *mut u8, old_size: usize, align: usize, new_size: usize) -> *mut u8;
    #[rustc_allocator_nounwind]
    fn __rust_alloc_zeroed(size: usize, align: usize) -> *mut u8;
}

/// The global memory allocator.
///
/// This type implements the [`AllocRef`] trait by forwarding calls
/// to the allocator registered with the `#[global_allocator]` attribute
/// if there is one, or the `std` crate’s default.
///
/// Note: while this type is unstable, the functionality it provides can be
/// accessed through the [free functions in `alloc`](index.html#functions).
///
/// [`AllocRef`]: trait.AllocRef.html
#[unstable(feature = "allocator_api", issue = "32838")]
#[derive(Copy, Clone, Default, Debug)]
pub struct Global;

/// Allocate memory with the global allocator.
///
/// This function forwards calls to the [`GlobalAlloc::alloc`] method
/// of the allocator registered with the `#[global_allocator]` attribute
/// if there is one, or the `std` crate’s default.
///
/// This function is expected to be deprecated in favor of the `alloc` method
/// of the [`Global`] type when it and the [`AllocRef`] trait become stable.
///
/// # Safety
///
/// See [`GlobalAlloc::alloc`].
///
/// [`Global`]: struct.Global.html
/// [`AllocRef`]: trait.AllocRef.html
/// [`GlobalAlloc::alloc`]: trait.GlobalAlloc.html#tymethod.alloc
///
/// # Examples
///
/// ```
/// use std::alloc::{alloc, dealloc, Layout};
///
/// unsafe {
///     let layout = Layout::new::<u16>();
///     let ptr = alloc(layout);
///
///     *(ptr as *mut u16) = 42;
///     assert_eq!(*(ptr as *mut u16), 42);
///
///     dealloc(ptr, layout);
/// }
/// ```
#[stable(feature = "global_alloc", since = "1.28.0")]
#[inline]
pub unsafe fn alloc(layout: Layout) -> *mut u8 {
    unsafe { __rust_alloc(layout.size(), layout.align()) }
}

/// Deallocate memory with the global allocator.
///
/// This function forwards calls to the [`GlobalAlloc::dealloc`] method
/// of the allocator registered with the `#[global_allocator]` attribute
/// if there is one, or the `std` crate’s default.
///
/// This function is expected to be deprecated in favor of the `dealloc` method
/// of the [`Global`] type when it and the [`AllocRef`] trait become stable.
///
/// # Safety
///
/// See [`GlobalAlloc::dealloc`].
///
/// [`Global`]: struct.Global.html
/// [`AllocRef`]: trait.AllocRef.html
/// [`GlobalAlloc::dealloc`]: trait.GlobalAlloc.html#tymethod.dealloc
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
/// of the [`Global`] type when it and the [`AllocRef`] trait become stable.
///
/// # Safety
///
/// See [`GlobalAlloc::realloc`].
///
/// [`Global`]: struct.Global.html
/// [`AllocRef`]: trait.AllocRef.html
/// [`GlobalAlloc::realloc`]: trait.GlobalAlloc.html#method.realloc
#[stable(feature = "global_alloc", since = "1.28.0")]
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
/// of the [`Global`] type when it and the [`AllocRef`] trait become stable.
///
/// # Safety
///
/// See [`GlobalAlloc::alloc_zeroed`].
///
/// [`Global`]: struct.Global.html
/// [`AllocRef`]: trait.AllocRef.html
/// [`GlobalAlloc::alloc_zeroed`]: trait.GlobalAlloc.html#method.alloc_zeroed
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
#[inline]
pub unsafe fn alloc_zeroed(layout: Layout) -> *mut u8 {
    unsafe { __rust_alloc_zeroed(layout.size(), layout.align()) }
}

#[unstable(feature = "allocator_api", issue = "32838")]
unsafe impl AllocRef for Global {
    #[inline]
    fn alloc(&mut self, layout: Layout) -> Result<MemoryBlock, AllocErr> {
        let size = layout.size();
        let ptr = if size == 0 {
            layout.dangling()
        } else {
            // SAFETY: `layout` is non-zero in size,
            unsafe { NonNull::new(alloc(layout)).ok_or(AllocErr)? }
        };
        Ok(MemoryBlock { ptr, size })
    }

    #[inline]
    fn alloc_zeroed(&mut self, layout: Layout) -> Result<MemoryBlock, AllocErr> {
        let size = layout.size();
        let ptr = if size == 0 {
            layout.dangling()
        } else {
            // SAFETY: `layout` is non-zero in size,
            unsafe { NonNull::new(alloc_zeroed(layout)).ok_or(AllocErr)? }
        };
        Ok(MemoryBlock { ptr, size })
    }

    #[inline]
    unsafe fn dealloc(&mut self, ptr: NonNull<u8>, layout: Layout) {
        if layout.size() != 0 {
            // SAFETY: `layout` is non-zero in size,
            // other conditions must be upheld by the caller
            unsafe { dealloc(ptr.as_ptr(), layout) }
        }
    }

    #[inline]
    unsafe fn grow(
        &mut self,
        ptr: NonNull<u8>,
        layout: Layout,
        new_size: usize,
    ) -> Result<MemoryBlock, AllocErr> {
        debug_assert!(
            new_size >= layout.size(),
            "`new_size` must be greater than or equal to `layout.size()`"
        );

        // SAFETY: `new_size` must be non-zero, which is checked in the match expression.
        // Other conditions must be upheld by the caller
        unsafe {
            match layout.size() {
                old_size if old_size == new_size => Ok(MemoryBlock { ptr, size: new_size }),
                0 => self.alloc(Layout::from_size_align_unchecked(new_size, layout.align())),
                old_size => {
                    // `realloc` probably checks for `new_size > size` or something similar.
                    intrinsics::assume(new_size > old_size);
                    let raw_ptr = realloc(ptr.as_ptr(), layout, new_size);
                    let ptr = NonNull::new(raw_ptr).ok_or(AllocErr)?;
                    Ok(MemoryBlock { ptr, size: new_size })
                }
            }
        }
    }

    #[inline]
    unsafe fn grow_zeroed(
        &mut self,
        ptr: NonNull<u8>,
        layout: Layout,
        new_size: usize,
    ) -> Result<MemoryBlock, AllocErr> {
        debug_assert!(
            new_size >= layout.size(),
            "`new_size` must be greater than or equal to `layout.size()`"
        );

        // SAFETY: `new_size` must be non-zero, which is checked in the match expression.
        // Other conditions must be upheld by the caller
        unsafe {
            match layout.size() {
                old_size if old_size == new_size => Ok(MemoryBlock { ptr, size: new_size }),
                0 => self.alloc_zeroed(Layout::from_size_align_unchecked(new_size, layout.align())),
                old_size => {
                    // `realloc` probably checks for `new_size > size` or something similar.
                    intrinsics::assume(new_size > old_size);
                    let raw_ptr = realloc(ptr.as_ptr(), layout, new_size);
                    raw_ptr.add(old_size).write_bytes(0, new_size - old_size);
                    let ptr = NonNull::new(raw_ptr).ok_or(AllocErr)?;
                    Ok(MemoryBlock { ptr, size: new_size })
                }
            }
        }
    }

    #[inline]
    unsafe fn shrink(
        &mut self,
        ptr: NonNull<u8>,
        layout: Layout,
        new_size: usize,
    ) -> Result<MemoryBlock, AllocErr> {
        let old_size = layout.size();
        debug_assert!(
            new_size <= old_size,
            "`new_size` must be smaller than or equal to `layout.size()`"
        );

        let ptr = if new_size == old_size {
            ptr
        } else if new_size == 0 {
            // SAFETY: `layout` is non-zero in size as `old_size` != `new_size`
            // Other conditions must be upheld by the caller
            unsafe {
                self.dealloc(ptr, layout);
            }
            layout.dangling()
        } else {
            // SAFETY: new_size is not zero,
            // Other conditions must be upheld by the caller
            let raw_ptr = unsafe {
                // `realloc` probably checks for `new_size < old_size` or something similar.
                intrinsics::assume(new_size < old_size);
                realloc(ptr.as_ptr(), layout, new_size)
            };
            NonNull::new(raw_ptr).ok_or(AllocErr)?
        };

        Ok(MemoryBlock { ptr, size: new_size })
    }
}

/// The allocator for unique pointers.
// This function must not unwind. If it does, MIR codegen will fail.
#[cfg(not(test))]
#[lang = "exchange_malloc"]
#[inline]
unsafe fn exchange_malloc(size: usize, align: usize) -> *mut u8 {
    let layout = unsafe { Layout::from_size_align_unchecked(size, align) };
    match Global.alloc(layout) {
        Ok(memory) => memory.ptr.as_ptr(),
        Err(_) => handle_alloc_error(layout),
    }
}

#[cfg_attr(not(test), lang = "box_free")]
#[inline]
// This signature has to be the same as `Box`, otherwise an ICE will happen.
// When an additional parameter to `Box` is added (like `A: AllocRef`), this has to be added here as
// well.
// For example if `Box` is changed to  `struct Box<T: ?Sized, A: AllocRef>(Unique<T>, A)`,
// this function has to be changed to `fn box_free<T: ?Sized, A: AllocRef>(Unique<T>, A)` as well.
pub(crate) unsafe fn box_free<T: ?Sized>(ptr: Unique<T>) {
    unsafe {
        let size = size_of_val(ptr.as_ref());
        let align = min_align_of_val(ptr.as_ref());
        let layout = Layout::from_size_align_unchecked(size, align);
        Global.dealloc(ptr.cast().into(), layout)
    }
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
#[rustc_allocator_nounwind]
pub fn handle_alloc_error(layout: Layout) -> ! {
    extern "Rust" {
        #[lang = "oom"]
        fn oom_impl(layout: Layout) -> !;
    }
    unsafe { oom_impl(layout) }
}
