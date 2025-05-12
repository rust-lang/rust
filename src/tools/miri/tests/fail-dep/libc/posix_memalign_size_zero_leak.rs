//@ignore-target: windows # No posix_memalign on Windows

use std::ptr;

fn main() {
    let mut ptr: *mut libc::c_void = ptr::null_mut();
    let align = 64;
    let size = 0;
    let _ = unsafe { libc::posix_memalign(&mut ptr, align, size) }; //~ERROR: memory leak
}
