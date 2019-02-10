use crate::io;
use crate::mem;
use crate::sys::c;

pub fn hashmap_random_keys() -> (u64, u64) {
    let mut v = (0, 0);
    let ret = unsafe {
        c::RtlGenRandom(&mut v as *mut _ as *mut u8,
                        mem::size_of_val(&v) as c::ULONG)
    };
    if ret == 0 {
        panic!("couldn't generate random bytes: {}",
               io::Error::last_os_error());
    }
    return v
}
