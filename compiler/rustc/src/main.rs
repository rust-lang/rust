// A note about jemalloc: rustc uses jemalloc when built for CI and
// distribution. The obvious way to do this is with the `#[global_allocator]`
// mechanism. However, for complicated reasons (see
// https://github.com/rust-lang/rust/pull/81782#issuecomment-784438001 for some
// details) that mechanism doesn't work here. Also, we must use a consistent
// allocator across the rustc <-> llvm boundary, and `#[global_allocator]`
// wouldn't provide that.
//
// Instead, we use a lower-level mechanism. rustc is linked with jemalloc in a
// way such that jemalloc's implementation of `malloc`, `free`, etc., override
// the libc allocator's implementation. This means that Rust's `System`
// allocator, which calls `libc::malloc()` et al., is actually calling into
// jemalloc.
//
// A consequence of not using `GlobalAlloc` (and the `tikv-jemallocator` crate
// provides an impl of that trait, which is called `Jemalloc`) is that we
// cannot use the sized deallocation APIs (`sdallocx`) that jemalloc provides.
// It's unclear how much performance is lost because of this.
//
// As for the symbol overrides in `main` below: we're pulling in a static copy
// of jemalloc. We need to actually reference its symbols for it to get linked.
// The two crates we link to here, `std` and `rustc_driver`, are both dynamic
// libraries. So we must reference jemalloc symbols one way or another, because
// this file is the only object code in the rustc executable.
#[cfg(feature = "tikv-jemalloc-sys")]
use tikv_jemalloc_sys as jemalloc_sys;

fn main() {
    // See the comment at the top of this file for an explanation of this.
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
