#![allow(unused)]

use std::ffi::c_void;

use libc::{self, pthread_attr_t, pthread_t};

/// Spawn a thread using `pthread_create`, abort the process on any errors.
pub unsafe fn spawn_pthread(
    f: extern "C" fn(*mut c_void) -> *mut c_void,
    value: *mut c_void,
) -> pthread_t {
    let mut thread_id: pthread_t = 0;

    let attr: *const pthread_attr_t = std::ptr::null();

    if unsafe { libc::pthread_create(&raw mut thread_id, attr, f, value) } != 0 {
        std::process::abort();
    }
    thread_id
}

/// Unsafe because we do *not* check that `F` is `Send + 'static`.
/// That makes it much easier to write tests...
pub unsafe fn spawn_pthread_closure<F: FnOnce()>(f: F) -> pthread_t {
    let mut thread_id: pthread_t = 0;
    let attr: *const pthread_attr_t = std::ptr::null();
    let f = Box::new(f);
    extern "C" fn thread_func<F: FnOnce()>(f: *mut c_void) -> *mut c_void {
        let f = unsafe { Box::from_raw(f as *mut F) };
        f();
        std::ptr::null_mut()
    }
    if unsafe {
        libc::pthread_create(
            &raw mut thread_id,
            attr,
            thread_func::<F>,
            Box::into_raw(f) as *mut c_void,
        )
    } != 0
    {
        std::process::abort();
    }
    thread_id
}

// Join the given pthread, abort the process on any errors.
pub unsafe fn join_pthread(thread_id: pthread_t) {
    if unsafe { libc::pthread_join(thread_id, std::ptr::null_mut()) } != 0 {
        std::process::abort();
    }
}

/// Spawn `N` threads using `pthread_create` without any arguments, abort the process on any errors.
pub unsafe fn spawn_pthreads_no_params<const N: usize>(
    functions: [extern "C" fn(*mut c_void) -> *mut c_void; N],
) -> [pthread_t; N] {
    functions.map(|func| spawn_pthread(func, std::ptr::null_mut()))
}

// Join the `N` given pthreads, abort the process on any errors.
pub unsafe fn join_pthreads<const N: usize>(thread_ids: [pthread_t; N]) {
    let _ = thread_ids.map(|id| join_pthread(id));
}
