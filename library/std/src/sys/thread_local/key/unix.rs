use crate::mem;

pub type Key = libc::pthread_key_t;

#[inline]
pub fn create(dtor: Option<unsafe extern "C" fn(*mut u8)>) -> Key {
    let mut key = 0;
    assert_eq!(unsafe { libc::pthread_key_create(&mut key, mem::transmute(dtor)) }, 0);
    key
}

#[inline]
pub unsafe fn set(key: Key, value: *mut u8) {
    let r = unsafe { libc::pthread_setspecific(key, value as *mut _) };
    debug_assert_eq!(r, 0);
}

#[inline]
#[cfg(any(not(target_thread_local), test))]
pub unsafe fn get(key: Key) -> *mut u8 {
    unsafe { libc::pthread_getspecific(key) as *mut u8 }
}

#[inline]
pub unsafe fn destroy(key: Key) {
    let r = unsafe { libc::pthread_key_delete(key) };
    debug_assert_eq!(r, 0);
}
