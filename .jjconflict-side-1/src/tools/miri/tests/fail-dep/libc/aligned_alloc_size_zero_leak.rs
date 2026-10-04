//@ignore-target: windows # Windows does not support the standard C11 aligned_alloc.

fn main() {
    // libc doesn't have this function (https://github.com/rust-lang/libc/issues/3689),
    // so we declare it ourselves.
    extern "C" {
        fn aligned_alloc(alignment: libc::size_t, size: libc::size_t) -> *mut libc::c_void;
    }

    // Make sure even zero-sized allocations need to be freed.

    unsafe {
        aligned_alloc(2, 0); //~ERROR: memory leaked
    }
}
