//@ignore-target: windows # No pthreads on Windows
//@run-native

use std::{process, ptr, thread};

static mut STATE: i32 = 0;

unsafe extern "C" fn dtor(ptr: *mut libc::c_void) {
    assert_eq!(ptr.addr(), 43);

    // We expect to be called exactly once.
    if STATE == 0 {
        STATE = 1;
    } else {
        process::abort();
    }
}

fn main() {
    unsafe {
        let dtor = dtor as unsafe extern "C" fn(*mut libc::c_void);

        let mut key: libc::pthread_key_t = 0;
        assert_eq!(libc::pthread_key_create(&mut key, Some(dtor)), 0);
        let mut key2: libc::pthread_key_t = 0;
        assert_eq!(libc::pthread_key_create(&mut key2, Some(dtor)), 0);

        let r = libc::pthread_setspecific(key, ptr::without_provenance_mut(42));
        assert_eq!(r, 0);
        let ptr = libc::pthread_getspecific(key);
        assert_eq!(ptr.addr(), 42);

        let t = thread::spawn(move || {
            // Set the value the dtor will see.
            let r = libc::pthread_setspecific(key, ptr::without_provenance_mut(43));
            assert_eq!(r, 0);
        });
        t.join().unwrap();
        // This called the dtor.
        assert_eq!({ STATE }, 1);

        // A thread that does not set the key, does not call the dtor.
        thread::spawn(move || {}).join().unwrap();

        // Destroying the key means the dtor does not get called.
        let t = thread::spawn(move || {
            let r = libc::pthread_setspecific(key2, ptr::without_provenance_mut(44));
            assert_eq!(r, 0);
            let r = libc::pthread_key_delete(key2);
            assert_eq!(r, 0);
        });
        t.join().unwrap();

        // And when we exit, that also does not call the dtor.
        let t = thread::spawn(move || {
            let r = libc::pthread_setspecific(key, ptr::without_provenance_mut(45));
            assert_eq!(r, 0);

            std::process::exit(0);
        });
        t.join().unwrap();
    }
}
