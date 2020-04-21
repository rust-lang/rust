// ignore-windows: No libc on Windows
// compile-flags: -Zmiri-disable-isolation

#![feature(rustc_private)]

extern crate libc;

use std::cell::UnsafeCell;
use std::mem::{self, MaybeUninit};
use std::sync::Arc;
use std::thread;

struct Mutex {
    inner: UnsafeCell<libc::pthread_mutex_t>,
}

unsafe impl Sync for Mutex {}

impl std::fmt::Debug for Mutex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Mutex")
    }
}

struct Cond {
    inner: UnsafeCell<libc::pthread_cond_t>,
}

unsafe impl Sync for Cond {}

impl std::fmt::Debug for Cond {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Cond")
    }
}

unsafe fn create_cond_attr_monotonic() -> libc::pthread_condattr_t {
    let mut attr = MaybeUninit::<libc::pthread_condattr_t>::uninit();
    assert_eq!(libc::pthread_condattr_init(attr.as_mut_ptr()), 0);
    assert_eq!(libc::pthread_condattr_setclock(attr.as_mut_ptr(), libc::CLOCK_MONOTONIC), 0);
    attr.assume_init()
}

unsafe fn create_cond(attr: Option<libc::pthread_condattr_t>) -> Cond {
    let cond: Cond = mem::zeroed();
    if let Some(mut attr) = attr {
        assert_eq!(libc::pthread_cond_init(cond.inner.get() as *mut _, &attr as *const _), 0);
        assert_eq!(libc::pthread_condattr_destroy(&mut attr as *mut _), 0);
    } else {
        assert_eq!(libc::pthread_cond_init(cond.inner.get() as *mut _, 0 as *const _), 0);
    }
    cond
}

unsafe fn create_mutex() -> Mutex {
    mem::zeroed()
}

unsafe fn create_timeout(seconds: i64) -> libc::timespec {
    let mut now: libc::timespec = mem::zeroed();
    assert_eq!(libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut now), 0);
    libc::timespec { tv_sec: now.tv_sec + seconds, tv_nsec: now.tv_nsec }
}

fn test_pthread_condattr_t() {
    unsafe {
        let mut attr = create_cond_attr_monotonic();
        let mut clock_id = MaybeUninit::<libc::clockid_t>::uninit();
        assert_eq!(libc::pthread_condattr_getclock(&attr as *const _, clock_id.as_mut_ptr()), 0);
        assert_eq!(clock_id.assume_init(), libc::CLOCK_MONOTONIC);
        assert_eq!(libc::pthread_condattr_destroy(&mut attr as *mut _), 0);
    }
}

fn test_signal() {
    unsafe {
        let cond = Arc::new(create_cond(None));
        let mutex = Arc::new(create_mutex());

        assert_eq!(libc::pthread_mutex_lock(mutex.inner.get() as *mut _), 0);

        let spawn_mutex = Arc::clone(&mutex);
        let spawn_cond = Arc::clone(&cond);
        let handle = thread::spawn(move || {
            assert_eq!(libc::pthread_mutex_lock(spawn_mutex.inner.get() as *mut _), 0);
            assert_eq!(libc::pthread_cond_signal(spawn_cond.inner.get() as *mut _), 0);
            assert_eq!(libc::pthread_mutex_unlock(spawn_mutex.inner.get() as *mut _), 0);
        });

        assert_eq!(
            libc::pthread_cond_wait(cond.inner.get() as *mut _, mutex.inner.get() as *mut _),
            0
        );
        assert_eq!(libc::pthread_mutex_unlock(mutex.inner.get() as *mut _), 0);

        handle.join().unwrap();

        let mutex = Arc::try_unwrap(mutex).unwrap();
        assert_eq!(libc::pthread_mutex_destroy(mutex.inner.get() as *mut _), 0);
        let cond = Arc::try_unwrap(cond).unwrap();
        assert_eq!(libc::pthread_cond_destroy(cond.inner.get() as *mut _), 0);
    }
}

fn test_broadcast() {
    unsafe {
        let cond = Arc::new(create_cond(None));
        let mutex = Arc::new(create_mutex());

        assert_eq!(libc::pthread_mutex_lock(mutex.inner.get() as *mut _), 0);

        let spawn_mutex = Arc::clone(&mutex);
        let spawn_cond = Arc::clone(&cond);
        let handle = thread::spawn(move || {
            assert_eq!(libc::pthread_mutex_lock(spawn_mutex.inner.get() as *mut _), 0);
            assert_eq!(libc::pthread_cond_broadcast(spawn_cond.inner.get() as *mut _), 0);
            assert_eq!(libc::pthread_mutex_unlock(spawn_mutex.inner.get() as *mut _), 0);
        });

        assert_eq!(
            libc::pthread_cond_wait(cond.inner.get() as *mut _, mutex.inner.get() as *mut _),
            0
        );
        assert_eq!(libc::pthread_mutex_unlock(mutex.inner.get() as *mut _), 0);

        handle.join().unwrap();

        let mutex = Arc::try_unwrap(mutex).unwrap();
        assert_eq!(libc::pthread_mutex_destroy(mutex.inner.get() as *mut _), 0);
        let cond = Arc::try_unwrap(cond).unwrap();
        assert_eq!(libc::pthread_cond_destroy(cond.inner.get() as *mut _), 0);
    }
}

fn test_timed_wait_timeout() {
    unsafe {
        let attr = create_cond_attr_monotonic();
        let cond = create_cond(Some(attr));
        let mutex = create_mutex();
        let timeout = create_timeout(1);

        assert_eq!(libc::pthread_mutex_lock(mutex.inner.get() as *mut _), 0);
        assert_eq!(
            libc::pthread_cond_timedwait(
                cond.inner.get() as *mut _,
                mutex.inner.get() as *mut _,
                &timeout
            ),
            libc::ETIMEDOUT
        );
        assert_eq!(libc::pthread_mutex_unlock(mutex.inner.get() as *mut _), 0);
        assert_eq!(libc::pthread_mutex_destroy(mutex.inner.get() as *mut _), 0);
        assert_eq!(libc::pthread_cond_destroy(cond.inner.get() as *mut _), 0);
    }
}

fn test_timed_wait_notimeout() {
    unsafe {
        let attr = create_cond_attr_monotonic();
        let cond = Arc::new(create_cond(Some(attr)));
        let mutex = Arc::new(create_mutex());
        let timeout = create_timeout(100);

        assert_eq!(libc::pthread_mutex_lock(mutex.inner.get() as *mut _), 0);

        let spawn_mutex = Arc::clone(&mutex);
        let spawn_cond = Arc::clone(&cond);
        let handle = thread::spawn(move || {
            assert_eq!(libc::pthread_mutex_lock(spawn_mutex.inner.get() as *mut _), 0);
            assert_eq!(libc::pthread_cond_signal(spawn_cond.inner.get() as *mut _), 0);
            assert_eq!(libc::pthread_mutex_unlock(spawn_mutex.inner.get() as *mut _), 0);
        });

        assert_eq!(
            libc::pthread_cond_timedwait(
                cond.inner.get() as *mut _,
                mutex.inner.get() as *mut _,
                &timeout
            ),
            0
        );
        assert_eq!(libc::pthread_mutex_unlock(mutex.inner.get() as *mut _), 0);

        handle.join().unwrap();

        let mutex = Arc::try_unwrap(mutex).unwrap();
        assert_eq!(libc::pthread_mutex_destroy(mutex.inner.get() as *mut _), 0);
        let cond = Arc::try_unwrap(cond).unwrap();
        assert_eq!(libc::pthread_cond_destroy(cond.inner.get() as *mut _), 0);
    }
}

fn main() {
    test_pthread_condattr_t();
    test_signal();
    test_broadcast();
    test_timed_wait_timeout();
    test_timed_wait_notimeout();
}
