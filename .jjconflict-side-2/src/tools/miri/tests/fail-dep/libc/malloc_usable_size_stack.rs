//@only-target: linux android freebsd

fn main() {
    unsafe {
        // A stack variable is not managed by the C allocator.
        let mut x = 42;
        let p = (&raw mut x).cast::<libc::c_void>();
        libc::malloc_usable_size(p); //~ERROR: not managed by the C allocator
    }
}
