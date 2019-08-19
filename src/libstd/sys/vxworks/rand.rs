use crate::mem;
use crate::slice;

pub fn hashmap_random_keys() -> (u64, u64) {
    let mut v = (0, 0);
    unsafe {
        let view = slice::from_raw_parts_mut(&mut v as *mut _ as *mut u8,
                                             mem::size_of_val(&v));
        imp::fill_bytes(view);
    }
    return v
}

mod imp {
    use libc;
    use crate::io;

    extern "C" {
        fn randBytes (randBuf: *mut libc::c_uchar,
                      numOfBytes: libc::c_int) -> libc::c_int;
    }

    pub fn fill_bytes(v: &mut [u8]) {
        let ret = unsafe {
            randBytes(v.as_mut_ptr() as *mut libc::c_uchar, v.len() as libc::c_int)
        };
        if ret == -1 {
            panic!("couldn't generate random bytes: {}", io::Error::last_os_error());
        }
    }
}
