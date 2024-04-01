#![feature(unix_sigpipe)]
// We need this feature as it changes `dylib` linking behavior and allows us to link to `rustc_driver`.
#![feature(rustc_private)]

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
//
// NOTE: if you are reading this comment because you want to set a custom `global_allocator` for
// benchmarking, consider using the benchmarks in the `rustc-perf` collector suite instead:
// https://github.com/rust-lang/rustc-perf/blob/master/collector/README.md#profiling
//
// NOTE: if you are reading this comment because you want to replace jemalloc with another allocator
// to compare their performance, see
// https://github.com/rust-lang/rust/commit/b90cfc887c31c3e7a9e6d462e2464db1fe506175#diff-43914724af6e464c1da2171e4a9b6c7e607d5bc1203fa95c0ab85be4122605ef
// for an example of how to do so.

use std::os::raw::{c_char, c_int, c_void};

#[no_mangle]
unsafe extern "C" fn calloc(items: usize, size: usize) -> *mut c_void {
    fjall::c::calloc(items, size)
}

#[no_mangle]
unsafe extern "C" fn posix_memalign(ptr: *mut *mut c_void, size: usize, align: usize) -> c_int {
    fjall::c::posix_memalign(ptr, size, align)
}

#[no_mangle]
unsafe extern "C" fn aligned_alloc(size: usize, align: usize) -> *mut c_void {
    fjall::c::aligned_alloc(size, align)
}

#[no_mangle]
unsafe extern "C" fn malloc(size: usize) -> *mut c_void {
    fjall::c::malloc(size)
}

#[no_mangle]
unsafe extern "C" fn realloc(ptr: *mut c_void, size: usize) -> *mut c_void {
    fjall::c::realloc(ptr, size)
}

#[no_mangle]
unsafe extern "C" fn free(ptr: *mut c_void) {
    fjall::c::free(ptr);
}

#[no_mangle]
unsafe extern "C" fn strdup(ptr: *const c_char) -> *mut c_char {
    fjall::c::strdup(ptr)
}

#[unix_sigpipe = "sig_dfl"]
fn main() {
    // See the comment at the top of this file for an explanation of this.
    {
        #[used]
        static _F1: unsafe extern "C" fn(usize, usize) -> *mut c_void = calloc;
        #[used]
        static _F2: unsafe extern "C" fn(*mut *mut c_void, usize, usize) -> c_int = posix_memalign;
        #[used]
        static _F3: unsafe extern "C" fn(usize, usize) -> *mut c_void = aligned_alloc;
        #[used]
        static _F4: unsafe extern "C" fn(usize) -> *mut c_void = malloc;
        #[used]
        static _F5: unsafe extern "C" fn(*mut c_void, usize) -> *mut c_void = realloc;
        #[used]
        static _F6: unsafe extern "C" fn(*mut c_void) = free;
        #[used]
        static _F7: unsafe extern "C" fn(*const c_char) -> *mut c_char = strdup;
        /*
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
        }*/
    }

    rustc_driver::main()
}
