// A note about mimalloc: rustc uses mimalloc when built for CI and
// distribution. The obvious way to do this is with the `#[global_allocator]`
// mechanism. However, for complicated reasons (see
// https://github.com/rust-lang/rust/pull/81782#issuecomment-784438001 for some
// details) that mechanism doesn't work here. Also, we must use a consistent
// allocator across the rustc <-> llvm boundary, and `#[global_allocator]`
// wouldn't provide that.
//
// Instead, we use a lower-level mechanism. rustc is linked with mimalloc in a
// way such that mimalloc's implementation of `malloc`, `free`, etc., override
// the libc allocator's implementation. This means that Rust's `System`
// allocator, which calls `libc::malloc()` et al., is actually calling into
// mimalloc.
//
// As for the symbol overrides in `main` below: we're pulling in a static copy
// of mimalloc. We need to actually reference its symbols for it to get linked.
// The two crates we link to here, `std` and `rustc_driver`, are both dynamic
// libraries. So we must reference mimalloc symbols one way or another, because
// this file is the only object code in the rustc executable.

fn main() {
    // See the comment at the top of this file for an explanation of this.
    #[cfg(feature = "mimallocate-sys")]
    {
        use std::os::raw::{c_int, c_void};

        #[used]
        static _F1: unsafe extern "C" fn(usize, usize) -> *mut c_void = mimallocate_sys::mi_calloc;
        #[used]
        static _F2: unsafe extern "C" fn(*mut *mut c_void, usize, usize) -> c_int =
            mimallocate_sys::mi_posix_memalign;
        #[used]
        static _F3: unsafe extern "C" fn(usize, usize) -> *mut c_void =
            mimallocate_sys::mi_aligned_alloc;
        #[used]
        static _F4: unsafe extern "C" fn(usize) -> *mut c_void = mimallocate_sys::mi_malloc;
        #[used]
        static _F5: unsafe extern "C" fn(*mut c_void, usize) -> *mut c_void =
            mimallocate_sys::mi_realloc;
        #[used]
        static _F6: unsafe extern "C" fn(*mut c_void) = mimallocate_sys::mi_free;

        // On OSX, mimalloc doesn't directly override malloc/free, but instead
        // registers itself with the allocator's zone APIs in a ctor. However,
        // the linker doesn't seem to consider ctors as "used" when statically
        // linking, so we need to explicitly depend on the function.
        #[cfg(target_os = "macos")]
        #[used]
        static _F7: unsafe extern "C" fn() =
            mimallocate_sys::_mi_macos_override_malloc;
    }

    rustc_driver::set_sigpipe_handler();
    rustc_driver::main()
}
