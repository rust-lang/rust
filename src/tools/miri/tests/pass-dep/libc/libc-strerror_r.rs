//@ignore-target: windows # Supported only on unixes

fn main() {
    unsafe {
        let mut buf = vec![0u8; 32];
        assert_eq!(libc::strerror_r(libc::EPERM, buf.as_mut_ptr().cast(), buf.len()), 0);
        let mut buf2 = vec![0u8; 64];
        assert_eq!(libc::strerror_r(-1i32, buf2.as_mut_ptr().cast(), buf2.len()), 0);
        // This buffer is deliberately too small so this triggers ERANGE.
        let mut buf3 = vec![0u8; 2];
        assert_eq!(
            libc::strerror_r(libc::E2BIG, buf3.as_mut_ptr().cast(), buf3.len()),
            libc::ERANGE
        );
    }
}
