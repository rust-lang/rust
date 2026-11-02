//! Test the order in which pthread dtors are run relative to native thread exit callbacks.
//@only-target: linux darwin
//@ignore-target: android
//@run-native

use std::{ffi, mem, ptr};

pub type Key = libc::pthread_key_t;

pub unsafe fn create(dtor: Option<unsafe extern "C" fn(*mut u8)>) -> Key {
    let mut key = 0;
    assert_eq!(libc::pthread_key_create(&mut key, mem::transmute(dtor)), 0);
    key
}

pub unsafe fn set(key: Key, value: *mut u8) {
    let r = libc::pthread_setspecific(key, value as *mut _);
    assert_eq!(r, 0);
}

unsafe extern "C" fn dtor(_ptr: *mut u8) {
    eprintln!("pthread key dtor");
}

fn register_native<F>(f: F)
where
    F: FnOnce() + 'static,
{
    // This will receive the pointer passed into `_tlv_atexit`, which is the
    // original `f` but boxed up.
    unsafe extern "C" fn run<F>(ptr: *mut ffi::c_void)
    where
        F: FnOnce() + 'static,
    {
        let f = unsafe { Box::from_raw(ptr as *mut F) };
        f()
    }

    cfg_select! {
        target_os = "linux" => {
            extern "C" {
                #[allow(improper_ctypes)]
                static __dso_handle: ();

                fn __cxa_thread_atexit_impl(
                    dtor: unsafe extern "C" fn(*mut ffi::c_void),
                    arg: *mut ffi::c_void,
                    dso: *const ffi::c_void,
                ) -> ffi::c_int;
            }

            let ret = unsafe {
                __cxa_thread_atexit_impl(
                    run::<F>,
                    Box::into_raw(Box::new(f)) as *mut ffi::c_void,
                    &raw const __dso_handle as *const ffi::c_void,
                )
            };
            assert_eq!(ret, 0);
        }
        target_vendor = "apple" => {
            extern "C" {
                fn _tlv_atexit(dtor: unsafe extern "C" fn(*mut ffi::c_void), arg: *mut ffi::c_void);
            }
            unsafe { _tlv_atexit(run::<F>, Box::into_raw(Box::new(f)) as *mut ffi::c_void) };
        }
    }
}

fn main() {
    let t = std::thread::spawn(|| unsafe {
        let key = create(Some(dtor));
        set(key, ptr::without_provenance_mut(1));
        register_native(|| eprintln!("native thread atexit"));
    });
    t.join().unwrap()
}
