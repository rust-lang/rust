//@only-target: linux darwin # the only targets where we have thread_atexit handlers
//@ignore-target: android

use std::ffi;

#[track_caller]
fn register<F>(f: F)
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
    register(|| unsafe { libc::exit(0) })
    //~^ERROR: `exit` called more than once
}
