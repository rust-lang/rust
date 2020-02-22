#![feature(rustc_private)]

use std::sync::{Mutex, RwLock, TryLockError};

extern crate libc;

fn main() {
    test_mutex();
    #[cfg(not(target_os = "windows"))] // TODO: implement RwLock on Windows
    {
        test_rwlock_stdlib();
        test_rwlock_libc_init();
        test_rwlock_libc_static_initializer();
    }
}

fn test_mutex() {
    let m = Mutex::new(0);
    {
        let _guard = m.lock();
        assert!(m.try_lock().unwrap_err().would_block());
    }
    drop(m.try_lock().unwrap());
    drop(m);
}

#[cfg(not(target_os = "windows"))]
fn test_rwlock_stdlib() {
    let rw = RwLock::new(0);
    {
        let _read_guard = rw.read().unwrap();
        drop(rw.read().unwrap());
        drop(rw.try_read().unwrap());
        assert!(rw.try_write().unwrap_err().would_block());
    }

    {
        let _write_guard = rw.write().unwrap();
        assert!(rw.try_read().unwrap_err().would_block());
        assert!(rw.try_write().unwrap_err().would_block());
    }
}

// need to go a layer deeper and test the behavior of libc functions, because
// std::sys::unix::rwlock::RWLock keeps track of write_locked and num_readers

#[cfg(not(target_os = "windows"))]
fn test_rwlock_libc_init() {
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
}

#[cfg(not(target_os = "windows"))]
fn test_rwlock_libc_static_initializer() {
    let rw = std::cell::UnsafeCell::new(libc::PTHREAD_RWLOCK_INITIALIZER);
    unsafe {
        assert_eq!(libc::pthread_rwlock_rdlock(rw.get()), 0);
        assert_eq!(libc::pthread_rwlock_rdlock(rw.get()), 0);
        assert_eq!(libc::pthread_rwlock_unlock(rw.get()), 0);
        assert_eq!(libc::pthread_rwlock_tryrdlock(rw.get()), 0);
        assert_eq!(libc::pthread_rwlock_unlock(rw.get()), 0);
        assert_eq!(libc::pthread_rwlock_trywrlock(rw.get()), libc::EBUSY);
        assert_eq!(libc::pthread_rwlock_unlock(rw.get()), 0);

        assert_eq!(libc::pthread_rwlock_wrlock(rw.get()), 0);
        assert_eq!(libc::pthread_rwlock_tryrdlock(rw.get()), libc::EBUSY);
        assert_eq!(libc::pthread_rwlock_trywrlock(rw.get()), libc::EBUSY);
        assert_eq!(libc::pthread_rwlock_unlock(rw.get()), 0);

        assert_eq!(libc::pthread_rwlock_destroy(rw.get()), 0);
    }
}

trait TryLockErrorExt<T> {
    fn would_block(&self) -> bool;
}

impl<T> TryLockErrorExt<T> for TryLockError<T> {
    fn would_block(&self) -> bool {
        match self {
            TryLockError::WouldBlock => true,
            TryLockError::Poisoned(_) => false,
        }
    }
}
