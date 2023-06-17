//@compile-flags: -Zmiri-disable-isolation
//@ignore-target-windows: No libc on Windows

#![feature(rustc_private)]

fn main() {
    unsafe {
        let ptr = libc::mmap(
            std::ptr::null_mut(),
            4096,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
            -1,
            0,
        );
        libc::munmap(ptr, 4096);
        let _x = *(ptr as *mut u8); //~ ERROR: was dereferenced after this allocation got freed
    }
}
