//! The man pages for mmap/munmap suggest that it is possible to partly unmap a previously-mapped
//! region of address space, but to LLVM that would be partial deallocation, which LLVM does not
//! support. So even though the man pages say this sort of use is possible, we must report UB.
//@ignore-target: windows # No mmap on Windows
//@normalize-stderr-test: "size [0-9]+ and alignment" -> "size SIZE and alignment"

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
        //~^ ERROR: Undefined Behavior
    }
}
