use crate::ptr;

pub fn fill_bytes(bytes: &mut [u8]) {
    let mib = [libc::CTL_KERN, libc::KERN_ARND];
    for chunk in bytes.chunks_mut(256) {
        let mut len = chunk.len();
        let ret = unsafe {
            libc::sysctl(
                mib.as_ptr(),
                mib.len() as libc::c_uint,
                chunk.as_mut_ptr().cast(),
                &mut len,
                ptr::null(),
                0,
            )
        };
        assert!(ret != -1 && len == chunk.len(), "failed to generate random data");
    }
}
