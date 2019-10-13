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
    use core::sync::atomic::{AtomicBool, Ordering::Relaxed};

    pub fn fill_bytes(v: &mut [u8]) {
        static RNG_INIT: AtomicBool = AtomicBool::new(false);
        while !RNG_INIT.load(Relaxed) {
            let ret = unsafe { libc::randSecure() };
            if ret < 0 {
                panic!("couldn't generate random bytes: {}", io::Error::last_os_error());
            } else if ret > 0 {
                RNG_INIT.store(true, Relaxed);
                break;
            }
            unsafe { libc::usleep(10) };
        }
        let ret = unsafe {
            libc::randABytes(v.as_mut_ptr() as *mut libc::c_uchar, v.len() as libc::c_int)
        };
        if ret < 0 {
            panic!("couldn't generate random bytes: {}", io::Error::last_os_error());
        }
    }
}
