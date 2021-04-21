use crate::io;
use crate::mem;
use crate::sys::c;

pub fn hashmap_random_keys() -> (u64, u64) {
    use crate::ptr;

    let mut v = (0, 0);
    let ret = unsafe {
        c::BCryptGenRandom(
            ptr::null_mut(),
            &mut v as *mut _ as *mut u8,
            mem::size_of_val(&v) as c::ULONG,
            c::BCRYPT_USE_SYSTEM_PREFERRED_RNG,
        )
    };
    if ret != 0 {
        panic!("couldn't generate random bytes: {}", io::Error::last_os_error());
    }
    return v;
}
