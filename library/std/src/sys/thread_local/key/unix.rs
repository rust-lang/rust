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

pub(crate) type Key = libc::pthread_key_t;

#[inline]
pub(crate) fn create(dtor: Option<unsafe extern "C" fn(*mut u8)>) -> Key {
    let mut key = 0;
    if unsafe { libc::pthread_key_create(&mut key, mem::transmute(dtor)) } != 0 {
        rtabort!("out of TLS keys");
    }
    key
}

#[cold]
fn fail() -> ! {
    rtabort!("Unexpected TLS failure")
}

#[inline]
pub(crate) unsafe fn set(key: Key, value: *mut u8) {
    let r = unsafe { libc::pthread_setspecific(key, value as *mut _) };
    // May happen on memory exhaustion
    if r != 0 {
        fail()
    }
}

#[inline]
#[cfg(any(not(target_thread_local), test))]
pub(crate) unsafe fn get(key: Key) -> *mut u8 {
    unsafe { libc::pthread_getspecific(key) as *mut u8 }
}

#[inline]
pub(crate) unsafe fn destroy(key: Key) {
    let r = unsafe { libc::pthread_key_delete(key) };
    // only documented error is for invalid keys
    if r != 0 {
        fail()
    }
}
