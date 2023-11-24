//@ignore-target-windows: No libc on Windows

fn main() {
    test_mutex_libc_init_recursive();
    test_mutex_libc_init_normal();
    test_mutex_libc_init_errorcheck();
    test_rwlock_libc_static_initializer();

    #[cfg(target_os = "linux")]
    test_mutex_libc_static_initializer_recursive();
}

fn test_mutex_libc_init_recursive() {
    unsafe {
        let mut attr: libc::pthread_mutexattr_t = std::mem::zeroed();
        assert_eq!(libc::pthread_mutexattr_init(&mut attr as *mut _), 0);
        assert_eq!(
            libc::pthread_mutexattr_settype(&mut attr as *mut _, libc::PTHREAD_MUTEX_RECURSIVE),
            0,
        );
        let mut mutex: libc::pthread_mutex_t = std::mem::zeroed();
        assert_eq!(libc::pthread_mutex_init(&mut mutex as *mut _, &mut attr as *mut _), 0);
        assert_eq!(libc::pthread_mutex_lock(&mut mutex as *mut _), 0);
        assert_eq!(libc::pthread_mutex_trylock(&mut mutex as *mut _), 0);
        assert_eq!(libc::pthread_mutex_unlock(&mut mutex as *mut _), 0);
        assert_eq!(libc::pthread_mutex_unlock(&mut mutex as *mut _), 0);
        assert_eq!(libc::pthread_mutex_trylock(&mut mutex as *mut _), 0);
        assert_eq!(libc::pthread_mutex_lock(&mut mutex as *mut _), 0);
        assert_eq!(libc::pthread_mutex_unlock(&mut mutex as *mut _), 0);
        assert_eq!(libc::pthread_mutex_unlock(&mut mutex as *mut _), 0);
        assert_eq!(libc::pthread_mutex_unlock(&mut mutex as *mut _), libc::EPERM);
        assert_eq!(libc::pthread_mutex_destroy(&mut mutex as *mut _), 0);
        assert_eq!(libc::pthread_mutexattr_destroy(&mut attr as *mut _), 0);
    }
}

fn test_mutex_libc_init_normal() {
    unsafe {
        let mut mutexattr: libc::pthread_mutexattr_t = std::mem::zeroed();
        assert_eq!(
            libc::pthread_mutexattr_settype(&mut mutexattr as *mut _, 0x12345678),
            libc::EINVAL,
        );
        assert_eq!(
            libc::pthread_mutexattr_settype(&mut mutexattr as *mut _, libc::PTHREAD_MUTEX_NORMAL),
            0,
        );
        let mut mutex: libc::pthread_mutex_t = std::mem::zeroed();
        assert_eq!(libc::pthread_mutex_init(&mut mutex as *mut _, &mutexattr as *const _), 0);
        assert_eq!(libc::pthread_mutex_lock(&mut mutex as *mut _), 0);
        assert_eq!(libc::pthread_mutex_trylock(&mut mutex as *mut _), libc::EBUSY);
        assert_eq!(libc::pthread_mutex_unlock(&mut mutex as *mut _), 0);
        assert_eq!(libc::pthread_mutex_trylock(&mut mutex as *mut _), 0);
        assert_eq!(libc::pthread_mutex_unlock(&mut mutex as *mut _), 0);
        assert_eq!(libc::pthread_mutex_destroy(&mut mutex as *mut _), 0);
    }
}

fn test_mutex_libc_init_errorcheck() {
    unsafe {
        let mut mutexattr: libc::pthread_mutexattr_t = std::mem::zeroed();
        assert_eq!(
            libc::pthread_mutexattr_settype(
                &mut mutexattr as *mut _,
                libc::PTHREAD_MUTEX_ERRORCHECK,
            ),
            0,
        );
        let mut mutex: libc::pthread_mutex_t = std::mem::zeroed();
        assert_eq!(libc::pthread_mutex_init(&mut mutex as *mut _, &mutexattr as *const _), 0);
        assert_eq!(libc::pthread_mutex_lock(&mut mutex as *mut _), 0);
        assert_eq!(libc::pthread_mutex_trylock(&mut mutex as *mut _), libc::EBUSY);
        assert_eq!(libc::pthread_mutex_lock(&mut mutex as *mut _), libc::EDEADLK);
        assert_eq!(libc::pthread_mutex_unlock(&mut mutex as *mut _), 0);
        assert_eq!(libc::pthread_mutex_trylock(&mut mutex as *mut _), 0);
        assert_eq!(libc::pthread_mutex_unlock(&mut mutex as *mut _), 0);
        assert_eq!(libc::pthread_mutex_unlock(&mut mutex as *mut _), libc::EPERM);
        assert_eq!(libc::pthread_mutex_destroy(&mut mutex as *mut _), 0);
    }
}

// Only linux provides PTHREAD_RECURSIVE_MUTEX_INITIALIZER_NP,
// libc for macOS just has the default PTHREAD_MUTEX_INITIALIZER.
#[cfg(target_os = "linux")]
fn test_mutex_libc_static_initializer_recursive() {
    let mutex = std::cell::UnsafeCell::new(libc::PTHREAD_RECURSIVE_MUTEX_INITIALIZER_NP);
    unsafe {
        assert_eq!(libc::pthread_mutex_lock(mutex.get()), 0);
        assert_eq!(libc::pthread_mutex_trylock(mutex.get()), 0);
        assert_eq!(libc::pthread_mutex_unlock(mutex.get()), 0);
        assert_eq!(libc::pthread_mutex_unlock(mutex.get()), 0);
        assert_eq!(libc::pthread_mutex_trylock(mutex.get()), 0);
        assert_eq!(libc::pthread_mutex_lock(mutex.get()), 0);
        assert_eq!(libc::pthread_mutex_unlock(mutex.get()), 0);
        assert_eq!(libc::pthread_mutex_unlock(mutex.get()), 0);
        assert_eq!(libc::pthread_mutex_unlock(mutex.get()), libc::EPERM);
        assert_eq!(libc::pthread_mutex_destroy(mutex.get()), 0);
    }
}

// Testing the behavior of std::sync::RwLock does not fully exercise the pthread rwlock shims, we
// need to go a layer deeper and test the behavior of the libc functions, because
// std::sys::unix::rwlock::RWLock itself keeps track of write_locked and num_readers.
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

        assert_eq!(libc::pthread_rwlock_trywrlock(rw.get()), 0);
        assert_eq!(libc::pthread_rwlock_tryrdlock(rw.get()), libc::EBUSY);
        assert_eq!(libc::pthread_rwlock_trywrlock(rw.get()), libc::EBUSY);
        assert_eq!(libc::pthread_rwlock_unlock(rw.get()), 0);

        assert_eq!(libc::pthread_rwlock_destroy(rw.get()), 0);
    }
}
