// Configure jemalloc as the `global_allocator` when configured. This is
// so that we use the sized deallocation apis jemalloc provides
// (namely `sdallocx`).
//
// The symbol overrides documented below are also performed so that we can
// ensure that we use a consistent allocator across the rustc <-> llvm boundary
#[cfg(feature = "jemalloc")]
#[global_allocator]
static ALLOC: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

#[cfg(feature = "tikv-jemalloc-sys")]
use tikv_jemalloc_sys as jemalloc_sys;

use std::alloc::{GlobalAlloc, Layout};
use tikv_jemallocator::Jemalloc;

// XXX: these declarations make rustc hook into tikv-jemallocator, which means
// jemalloc's sized deallocation path (`sdallocx`) gets used. Without them,
// rustc hooks in at a lower level, via tikv-jemalloc-sys, which uses the
// vanilla `free` path. In theory, the `sdallocx` path is faster.
//
// Note that this is a hack that works on Linux, but probably doesn't work on
// other platforms. It's just for testing purposes.

#[no_mangle]
pub unsafe extern "C" fn __rust_alloc(size: usize, align: usize) -> *mut u8 {
    Jemalloc.alloc(Layout::from_size_align_unchecked(size, align))
}

#[no_mangle]
pub unsafe extern "C" fn __rust_alloc_zeroed(size: usize, align: usize) -> *mut u8 {
    Jemalloc.alloc_zeroed(Layout::from_size_align_unchecked(size, align))
}

#[no_mangle]
pub unsafe extern "C" fn __rust_dealloc(ptr: *mut u8, size: usize, align: usize) {
    Jemalloc.dealloc(ptr, Layout::from_size_align_unchecked(size, align))
}

#[no_mangle]
pub unsafe extern "C" fn __rust_realloc(
    ptr: *mut u8,
    old_size: usize,
    align: usize,
    new_size: usize,
) -> *mut u8 {
    Jemalloc.realloc(ptr, Layout::from_size_align_unchecked(old_size, align), new_size)
}

fn main() {
    // Pull in jemalloc when enabled.
    //
    // Note that we're pulling in a static copy of jemalloc which means that to
    // pull it in we need to actually reference its symbols for it to get
    // linked. The two crates we link to here, std and rustc_driver, are both
    // dynamic libraries. That means to pull in jemalloc we actually need to
    // reference allocation symbols one way or another (as this file is the only
    // object code in the rustc executable).
    #[cfg(feature = "tikv-jemalloc-sys")]
    {
        use std::os::raw::{c_int, c_void};

        #[used]
        static _F1: unsafe extern "C" fn(usize, usize) -> *mut c_void = jemalloc_sys::calloc;
        #[used]
        static _F2: unsafe extern "C" fn(*mut *mut c_void, usize, usize) -> c_int =
            jemalloc_sys::posix_memalign;
        #[used]
        static _F3: unsafe extern "C" fn(usize, usize) -> *mut c_void = jemalloc_sys::aligned_alloc;
        #[used]
        static _F4: unsafe extern "C" fn(usize) -> *mut c_void = jemalloc_sys::malloc;
        #[used]
        static _F5: unsafe extern "C" fn(*mut c_void, usize) -> *mut c_void = jemalloc_sys::realloc;
        #[used]
        static _F6: unsafe extern "C" fn(*mut c_void) = jemalloc_sys::free;

        // On OSX, jemalloc doesn't directly override malloc/free, but instead
        // registers itself with the allocator's zone APIs in a ctor. However,
        // the linker doesn't seem to consider ctors as "used" when statically
        // linking, so we need to explicitly depend on the function.
        #[cfg(target_os = "macos")]
        {
            extern "C" {
                fn _rjem_je_zone_register();
            }

            #[used]
            static _F7: unsafe extern "C" fn() = _rjem_je_zone_register;
        }
    }

    rustc_driver::set_sigpipe_handler();
    rustc_driver::main()
}
