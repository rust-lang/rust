// Just instantiate some data structures to make sure we got all their foreign items covered.
// Requires full MIR on Windows.

#![feature(rustc_private)]

use std::sync;

extern crate libc;

fn main() {
    let m = sync::Mutex::new(0);
    {
        let _guard = m.lock();
        let try_lock_error = m.try_lock().unwrap_err();
        if let sync::TryLockError::Poisoned(e) = try_lock_error {
            panic!("{}", e);
        }
    }
    drop(m.try_lock().unwrap());
    drop(m);

    #[cfg(not(target_os = "windows"))] // TODO: implement RwLock on Windows
    {
        let rw = sync::RwLock::new(0);
        {
            let _read_guard = rw.read().unwrap();
            drop(rw.read().unwrap());
            drop(rw.try_read().unwrap());
            let try_lock_error = rw.try_write().unwrap_err();
            if let sync::TryLockError::Poisoned(e) = try_lock_error {
                panic!("{}", e);
            }
        }

        {
            let _write_guard = rw.write().unwrap();
            let try_lock_error = rw.try_read().unwrap_err();
            if let sync::TryLockError::Poisoned(e) = try_lock_error {
                panic!("{}", e);
            }
            let try_lock_error = rw.try_write().unwrap_err();
            if let sync::TryLockError::Poisoned(e) = try_lock_error {
                panic!("{}", e);
            }
        }

        // need to go a layer deeper and test the behavior of libc functions, because
        // std::sys::unix::rwlock::RWLock keeps track of write_locked and num_readers

        unsafe {
            let mut mutex: libc::pthread_mutex_t = std::mem::zeroed();
            assert_eq!(libc::pthread_mutex_init(&mut mutex as *mut _, std::ptr::null_mut()), 0);
            assert_eq!(libc::pthread_mutex_lock(&mut mutex as *mut _), 0);
            assert_eq!(libc::pthread_mutex_trylock(&mut mutex as *mut _), libc::EBUSY);
            assert_eq!(libc::pthread_mutex_unlock(&mut mutex as *mut _), 0);
            assert_eq!(libc::pthread_mutex_trylock(&mut mutex as *mut _), 0);
            assert_eq!(libc::pthread_mutex_unlock(&mut mutex as *mut _), 0);
            assert_eq!(libc::pthread_mutex_destroy(&mut mutex as *mut _), 0);
        }

        unsafe {
            let mut rw: libc::pthread_rwlock_t = std::mem::zeroed();
            assert_eq!(libc::pthread_rwlock_init(&mut rw as *mut _, std::ptr::null_mut()), 0);

            assert_eq!(libc::pthread_rwlock_rdlock(&mut rw as *mut _), 0);
            assert_eq!(libc::pthread_rwlock_rdlock(&mut rw as *mut _), 0);
            assert_eq!(libc::pthread_rwlock_unlock(&mut rw as *mut _), 0);
            assert_eq!(libc::pthread_rwlock_tryrdlock(&mut rw as *mut _), 0);
            assert_eq!(libc::pthread_rwlock_unlock(&mut rw as *mut _), 0);
            assert_eq!(libc::pthread_rwlock_trywrlock(&mut rw as *mut _), libc::EBUSY);
            assert_eq!(libc::pthread_rwlock_unlock(&mut rw as *mut _), 0);

            assert_eq!(libc::pthread_rwlock_wrlock(&mut rw as *mut _), 0);
            assert_eq!(libc::pthread_rwlock_tryrdlock(&mut rw as *mut _), libc::EBUSY);
            assert_eq!(libc::pthread_rwlock_trywrlock(&mut rw as *mut _), libc::EBUSY);
            assert_eq!(libc::pthread_rwlock_unlock(&mut rw as *mut _), 0);

            assert_eq!(libc::pthread_rwlock_destroy(&mut rw as *mut _), 0);
        }
    }
}
