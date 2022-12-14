//@ignore-target-windows: No libc on Windows

use core::slice;

fn main() {
    unsafe {
        let p1 = libc::calloc(0, 0);
        assert!(p1.is_null());

        let p2 = libc::calloc(20, 0);
        assert!(p2.is_null());

        let p3 = libc::calloc(0, 20);
        assert!(p3.is_null());

        let p4 = libc::calloc(4, 8);
        assert!(!p4.is_null());
        let slice = slice::from_raw_parts(p4 as *const u8, 4 * 8);
        assert_eq!(&slice, &[0_u8; 4 * 8]);
        libc::free(p4);
    }
}
