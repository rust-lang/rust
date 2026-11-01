//@only-target: linux android freebsd

fn main() {
    unsafe {
        // A Rust heap allocation is not managed by the C allocator.
        let b = Box::new(42);
        let p = Box::into_raw(b).cast::<libc::c_void>();
        libc::malloc_usable_size(p); //~ERROR: not managed by the C allocator
    }
}
