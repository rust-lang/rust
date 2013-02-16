// xfail-test not a test

use core::libc::{c_uint, c_int, c_void};
use core::ptr::null;

#[cfg(unix)]
pub type Key = pthread_key_t;

#[cfg(unix)]
pub unsafe fn create(key: &mut Key) {
    unsafe { assert 0 == pthread_key_create(key, null()); }
}

#[cfg(unix)]
pub unsafe fn set(key: Key, value: *mut c_void) {
    unsafe { assert 0 == pthread_setspecific(key, value); }
}

#[cfg(unix)]
pub unsafe fn get(key: Key) -> *mut c_void {
    unsafe { pthread_getspecific(key) }
}

#[cfg(unix)]
type pthread_key_t = c_uint;

#[cfg(unix)]
extern {
    fn pthread_key_create(key: *mut pthread_key_t, dtor: *u8) -> c_int;
    fn pthread_setspecific(key: pthread_key_t, value: *mut c_void) -> c_int;
    fn pthread_getspecific(key: pthread_key_t) -> *mut c_void;
}

#[test]
fn tls_smoke_test() {
    use core::cast::transmute;
    unsafe {
        let mut key = 0;
        let value = ~20;
        create(&mut key);
        set(key, transmute(value));
        let value: ~int = transmute(get(key));
        assert value == ~20;
        let value = ~30;
        set(key, transmute(value));
        let value: ~int = transmute(get(key));
        assert value == ~30;
    }
}
