//! Test the Linux and macOS APIs for thread exit callbacks.
//@only-target: linux darwin
//@ignore-target: android
//@run-native
//@revisions: in_thread with_exit without_exit

use std::{ffi, thread};

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

fn the_test() {
    thread::spawn(|| {
        register(|| println!("dtor 2"));
        register(|| println!("dtor 1"));
        println!("exiting thread");
    })
    .join()
    .unwrap();

    register(|| println!("dtor 5"));
    register(|| {
        println!("registering dtor in dtor 3");
        register(|| println!("dtor 4"));
    });

    println!("exiting main");
    if cfg!(with_exit) {
        std::process::exit(0);
    }
}

fn main() {
    if cfg!(in_thread) {
        return std::thread::spawn(the_test).join().unwrap();
    }

    the_test();

    if cfg!(with_exit) {
        std::process::exit(0);
    }
}
