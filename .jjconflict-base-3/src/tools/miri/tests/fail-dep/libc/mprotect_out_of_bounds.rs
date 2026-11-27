//@compile-flags: -Zmiri-disable-isolation
//@ignore-target: windows # No mmap on Windows
//@normalize-stderr-test: "only .*? bytes" -> "only SIZE bytes"

fn main() {
    unsafe {
        let page_size = page_size::get();
        let ptr = libc::mmap(
            std::ptr::null_mut(),
            page_size,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
            -1,
            0,
        );
        assert!(!ptr.is_null());

        libc::mprotect(ptr, page_size + 1, libc::PROT_READ | libc::PROT_WRITE); //~ ERROR: `mprotect` called on out-of-bounds memory
    }
}
