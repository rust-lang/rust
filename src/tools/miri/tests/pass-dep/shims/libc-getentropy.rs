//@ignore-target-windows: no libc

use libc::getentropy;

fn main() {
    let mut buf1 = [0u8; 256];
    let mut buf2 = [0u8; 257];
    unsafe {
        assert_eq!(getentropy(buf1.as_mut_ptr() as *mut libc::c_void, buf1.len()), 0);
        assert_eq!(getentropy(buf2.as_mut_ptr() as *mut libc::c_void, buf2.len()), -1);
        assert_eq!(std::io::Error::last_os_error().raw_os_error().unwrap(), libc::EIO);
    }
}
