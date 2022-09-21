#![allow(dead_code)] // not used on all platforms

use crate::mem;

pub type Key = libc::pthread_key_t;

#[inline]
pub unsafe fn create(dtor: Option<unsafe extern "C" fn(*mut u8)>) -> Key {
    let dtor = mem::transmute(dtor);
    let mut key = 0;
    let r = libc::pthread_key_create(&mut key, dtor);
    assert_eq!(r, 0);

    // POSIX allows the key created here to be 0, but `StaticKey` relies
    // on using 0 as a sentinel value to check who won the race to set the
    // shared TLS key. As far as I know, there is no guaranteed value that
    // cannot be returned as a posix_key_create key, so there is no value
    // we can initialize the inner key with to prove that it has not yet
    // been set. Therefore, we use this small trick to ensure the returned
    // key is not zero.
    if key == 0 {
        let mut new = 0;
        // Only check the creation result after deleting the old key to avoid
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
