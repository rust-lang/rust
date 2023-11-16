//@ignore-target-windows: no libc
//@ignore-target-apple: no support (yet)

use core::ptr;

fn main() {
    unsafe {
        let mut p = libc::reallocarray(ptr::null_mut(), 4096, 2);
        assert!(!p.is_null());
        libc::free(p);
        p = libc::malloc(16);
        let r = libc::reallocarray(p, 2, 32);
        assert!(!r.is_null());
        libc::free(r);
    }
}
