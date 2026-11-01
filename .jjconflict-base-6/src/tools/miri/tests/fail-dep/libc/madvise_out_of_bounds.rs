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

        libc::madvise(ptr, page_size + 1, libc::MADV_NORMAL); //~ ERROR: `madvise` called on out-of-bounds memory
    }
}
