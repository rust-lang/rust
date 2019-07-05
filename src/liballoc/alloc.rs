//! Memory allocation APIs

#![stable(feature = "alloc_module", since = "1.28.0")]

use core::intrinsics::{min_align_of_val, size_of_val};
use core::ptr::{NonNull, Unique};
use core::usize;

#[stable(feature = "alloc_module", since = "1.28.0")]
#[doc(inline)]
pub use core::alloc::*;

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
    fn __rust_realloc(ptr: *mut u8,
                      old_size: usize,
                      align: usize,
                      new_size: usize) -> *mut u8;
    #[rustc_allocator_nounwind]
    fn __rust_alloc_zeroed(size: usize, align: usize) -> *mut u8;
}

/// The global memory allocator.
///
/// This type implements the [`Alloc`] trait by forwarding calls
/// to the allocator registered with the `#[global_allocator]` attribute
/// if there is one, or the `std` crate’s default.
///
/// Note: while this type is unstable, the functionality it provides can be
/// accessed through the [free functions in `alloc`](index.html#functions).
///
/// [`Alloc`]: trait.Alloc.html
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
/// of the [`Global`] type when it and the [`Alloc`] trait become stable.
///
/// # Safety
///
/// See [`GlobalAlloc::alloc`].
///
/// [`Global`]: struct.Global.html
/// [`Alloc`]: trait.Alloc.html
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
    __rust_alloc(layout.size(), layout.align())
}

/// Deallocate memory with the global allocator.
///
/// This function forwards calls to the [`GlobalAlloc::dealloc`] method
/// of the allocator registered with the `#[global_allocator]` attribute
/// if there is one, or the `std` crate’s default.
///
/// This function is expected to be deprecated in favor of the `dealloc` method
/// of the [`Global`] type when it and the [`Alloc`] trait become stable.
///
/// # Safety
///
/// See [`GlobalAlloc::dealloc`].
///
/// [`Global`]: struct.Global.html
/// [`Alloc`]: trait.Alloc.html
/// [`GlobalAlloc::dealloc`]: trait.GlobalAlloc.html#tymethod.dealloc
#[stable(feature = "global_alloc", since = "1.28.0")]
#[inline]
pub unsafe fn dealloc(ptr: *mut u8, layout: Layout) {
    __rust_dealloc(ptr, layout.size(), layout.align())
}

/// Reallocate memory with the global allocator.
///
/// This function forwards calls to the [`GlobalAlloc::realloc`] method
/// of the allocator registered with the `#[global_allocator]` attribute
/// if there is one, or the `std` crate’s default.
///
/// This function is expected to be deprecated in favor of the `realloc` method
/// of the [`Global`] type when it and the [`Alloc`] trait become stable.
///
/// # Safety
///
/// See [`GlobalAlloc::realloc`].
///
/// [`Global`]: struct.Global.html
/// [`Alloc`]: trait.Alloc.html
/// [`GlobalAlloc::realloc`]: trait.GlobalAlloc.html#method.realloc
#[stable(feature = "global_alloc", since = "1.28.0")]
#[inline]
pub unsafe fn realloc(ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
    __rust_realloc(ptr, layout.size(), layout.align(), new_size)
}

/// Allocate zero-initialized memory with the global allocator.
///
/// This function forwards calls to the [`GlobalAlloc::alloc_zeroed`] method
/// of the allocator registered with the `#[global_allocator]` attribute
/// if there is one, or the `std` crate’s default.
///
/// This function is expected to be deprecated in favor of the `alloc_zeroed` method
/// of the [`Global`] type when it and the [`Alloc`] trait become stable.
///
/// # Safety
///
/// See [`GlobalAlloc::alloc_zeroed`].
///
/// [`Global`]: struct.Global.html
/// [`Alloc`]: trait.Alloc.html
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
    __rust_alloc_zeroed(layout.size(), layout.align())
}

#[unstable(feature = "allocator_api", issue = "32838")]
unsafe impl Alloc for Global {
    #[inline]
    unsafe fn alloc(&mut self, layout: Layout) -> Result<NonNull<u8>, AllocErr> {
        NonNull::new(alloc(layout)).ok_or(AllocErr)
    }

    #[inline]
    unsafe fn dealloc(&mut self, ptr: NonNull<u8>, layout: Layout) {
        dealloc(ptr.as_ptr(), layout)
    }

    #[inline]
    unsafe fn realloc(&mut self,
                      ptr: NonNull<u8>,
                      layout: Layout,
                      new_size: usize)
                      -> Result<NonNull<u8>, AllocErr>
    {
        NonNull::new(realloc(ptr.as_ptr(), layout, new_size)).ok_or(AllocErr)
    }

    #[inline]
    unsafe fn alloc_zeroed(&mut self, layout: Layout) -> Result<NonNull<u8>, AllocErr> {
        NonNull::new(alloc_zeroed(layout)).ok_or(AllocErr)
    }
}

/// The allocator for unique pointers.
// This function must not unwind. If it does, MIR codegen will fail.
#[cfg(not(test))]
#[lang = "exchange_malloc"]
#[inline]
unsafe fn exchange_malloc(size: usize, align: usize) -> *mut u8 {
    if size == 0 {
        align as *mut u8
    } else {
        let layout = Layout::from_size_align_unchecked(size, align);
        let ptr = alloc(layout);
        if !ptr.is_null() {
            ptr
        } else {
            handle_alloc_error(layout)
        }
    }
}

#[cfg_attr(not(test), lang = "box_free")]
#[inline]
pub(crate) unsafe fn box_free<T: ?Sized>(ptr: Unique<T>) {
    let ptr = ptr.as_ptr();
    let size = size_of_val(&*ptr);
    let align = min_align_of_val(&*ptr);
    // We do not allocate for Box<T> when T is ZST, so deallocation is also not necessary.
    if size != 0 {
        let layout = Layout::from_size_align_unchecked(size, align);
        dealloc(ptr as *mut u8, layout);
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
    #[allow(improper_ctypes)]
    extern "Rust" {
        #[lang = "oom"]
        fn oom_impl(layout: Layout) -> !;
    }
    unsafe { oom_impl(layout) }
}

#[cfg(test)]
mod tests {
    extern crate test;
    use test::Bencher;
    use crate::boxed::Box;
    use crate::alloc::{Global, Alloc, Layout, handle_alloc_error};

    #[test]
    fn allocate_zeroed() {
        unsafe {
            let layout = Layout::from_size_align(1024, 1).unwrap();
            let ptr = Global.alloc_zeroed(layout.clone())
                .unwrap_or_else(|_| handle_alloc_error(layout));

            let mut i = ptr.cast::<u8>().as_ptr();
            let end = i.add(layout.size());
            while i < end {
                assert_eq!(*i, 0);
                i = i.offset(1);
            }
            Global.dealloc(ptr, layout);
        }
    }

    #[bench]
    #[cfg(not(miri))] // Miri does not support benchmarks
    fn alloc_owned_small(b: &mut Bencher) {
        b.iter(|| {
            let _: Box<_> = box 10;
        })
    }
}
