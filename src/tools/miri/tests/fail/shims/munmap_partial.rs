//! Our mmap/munmap support is a thin wrapper over Interpcx::allocate_ptr. Since the underlying
//! layer has much more UB than munmap does, we need to be sure we throw an unsupported error here.
//@ignore-target-windows: No libc on Windows

fn main() {
    unsafe {
        let ptr = libc::mmap(
            std::ptr::null_mut(),
            page_size::get() * 2,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
            -1,
            0,
        );
        libc::munmap(ptr, 1);
        //~^ ERROR: unsupported operation
    }
}
