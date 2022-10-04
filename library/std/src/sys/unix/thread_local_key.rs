#![allow(dead_code)] // not used on all platforms

use crate::mem;

pub type Key = libc::pthread_key_t;

#[inline]
pub unsafe fn create(dtor: Option<unsafe extern "C" fn(*mut u8)>) -> Key {
    let dtor = mem::transmute(dtor);
    let mut key = 0;
    assert_eq!(libc::pthread_key_create(&mut key, dtor), 0);
    // POSIX allows the key created here to be 0, but `StaticKey` needs to
    // use 0 as a sentinel value to check who won the race to set the shared
    // TLS key. Therefore, we employ this small trick to avoid having to waste
    // the key value.
    if key == 0 {
        let mut new = 0;
        // Only check the results after the old key has been destroyed to avoid
        // leaking it.
        let r_c = libc::pthread_key_create(&mut new, dtor);
        let r_d = libc::pthread_key_delete(key);
        assert_eq!(r_c, 0);
        debug_assert_eq!(r_d, 0);
        key = new;
    }
    key
}

#[inline]
pub unsafe fn set(key: Key, value: *mut u8) {
    let r = libc::pthread_setspecific(key, value as *mut _);
    debug_assert_eq!(r, 0);
}

#[inline]
pub unsafe fn get(key: Key) -> *mut u8 {
    libc::pthread_getspecific(key) as *mut u8
}

#[inline]
pub unsafe fn destroy(key: Key) {
    let r = libc::pthread_key_delete(key);
    debug_assert_eq!(r, 0);
}

#[inline]
pub fn requires_synchronized_create() -> bool {
    false
}
