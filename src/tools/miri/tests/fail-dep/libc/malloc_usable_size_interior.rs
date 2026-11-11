//@only-target: linux android freebsd

fn main() {
    unsafe {
        // The pointer must point to the beginning of the block.
        let p = libc::malloc(1024);
        let mid = p.cast::<u8>().add(512).cast::<libc::c_void>();
        libc::malloc_usable_size(mid); //~ERROR: does not point to the beginning
    }
}
