use crate::mem;

// For WASI add a few symbols not in upstream `libc` just yet.
#[cfg(all(target_os = "wasi", target_env = "p1", target_feature = "atomics"))]
mod libc {
    use crate::ffi;

    #[allow(non_camel_case_types)]
    pub type pthread_key_t = ffi::c_uint;

    unsafe extern "C" {
        pub fn pthread_key_create(
            key: *mut pthread_key_t,
            destructor: unsafe extern "C" fn(*mut ffi::c_void),
        ) -> ffi::c_int;
        #[allow(dead_code)]
        pub fn pthread_getspecific(key: pthread_key_t) -> *mut ffi::c_void;
        pub fn pthread_setspecific(key: pthread_key_t, value: *const ffi::c_void) -> ffi::c_int;
        pub fn pthread_key_delete(key: pthread_key_t) -> ffi::c_int;
    }
}

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
