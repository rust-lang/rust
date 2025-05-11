//@ignore-target: windows # No pthreads on Windows
// We use `yield` to test specific interleavings, so disable automatic preemption.
//@compile-flags: -Zmiri-deterministic-concurrency
#![feature(sync_unsafe_cell)]

use std::cell::SyncUnsafeCell;
use std::mem::MaybeUninit;
use std::{mem, ptr, thread};

fn main() {
    test_mutex_libc_init_recursive();
    test_mutex_libc_init_normal();
    test_mutex_libc_init_errorcheck();
    test_rwlock_libc_static_initializer();
    #[cfg(target_os = "linux")]
    test_mutex_libc_static_initializer_recursive();

    check_mutex();
    check_rwlock_write();
    check_rwlock_read_no_deadlock();
    check_cond();
    check_condattr();
}

// We want to only use pthread APIs here for easier testing.
// So we can't use `thread::scope`. That means panics can lead
// to a failure to join threads which can lead to further issues,
// so let's turn such unwinding into aborts.
struct AbortOnDrop;
impl AbortOnDrop {
    fn defuse(self) {
        mem::forget(self);
    }
}
impl Drop for AbortOnDrop {
    fn drop(&mut self) {
        std::process::abort();
    }
}

fn test_mutex_libc_init_recursive() {
    unsafe {
        let mut attr: libc::pthread_mutexattr_t = mem::zeroed();
        assert_eq!(libc::pthread_mutexattr_init(&mut attr as *mut _), 0);
        assert_eq!(
            libc::pthread_mutexattr_settype(&mut attr as *mut _, libc::PTHREAD_MUTEX_RECURSIVE),
            0,
        );
        let mut mutex: libc::pthread_mutex_t = mem::zeroed();
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
        let mut mutexattr: libc::pthread_mutexattr_t = mem::zeroed();
        assert_eq!(
            libc::pthread_mutexattr_settype(&mut mutexattr as *mut _, 0x12345678),
            libc::EINVAL,
        );
        assert_eq!(
            libc::pthread_mutexattr_settype(&mut mutexattr as *mut _, libc::PTHREAD_MUTEX_NORMAL),
            0,
        );
        let mut mutex: libc::pthread_mutex_t = mem::zeroed();
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
        let mut mutexattr: libc::pthread_mutexattr_t = mem::zeroed();
        assert_eq!(
            libc::pthread_mutexattr_settype(
                &mut mutexattr as *mut _,
                libc::PTHREAD_MUTEX_ERRORCHECK,
            ),
            0,
        );
        let mut mutex: libc::pthread_mutex_t = mem::zeroed();
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

struct SendPtr<T> {
    ptr: *mut T,
}
unsafe impl<T> Send for SendPtr<T> {}
impl<T> Copy for SendPtr<T> {}
impl<T> Clone for SendPtr<T> {
    fn clone(&self) -> Self {
        *self
    }
}

fn check_mutex() {
    let bomb = AbortOnDrop;
    // Specifically *not* using `Arc` to make sure there is no synchronization apart from the mutex.
    unsafe {
        let data = SyncUnsafeCell::new((libc::PTHREAD_MUTEX_INITIALIZER, 0));
        let ptr = SendPtr { ptr: data.get() };
        let mut threads = Vec::new();

        for _ in 0..3 {
            let thread = thread::spawn(move || {
                let ptr = ptr; // circumvent per-field closure capture
                let mutexptr = ptr::addr_of_mut!((*ptr.ptr).0);
                assert_eq!(libc::pthread_mutex_lock(mutexptr), 0);
                thread::yield_now();
                (*ptr.ptr).1 += 1;
                assert_eq!(libc::pthread_mutex_unlock(mutexptr), 0);
            });
            threads.push(thread);
        }

        for thread in threads {
            thread.join().unwrap();
        }

        let mutexptr = ptr::addr_of_mut!((*ptr.ptr).0);
        assert_eq!(libc::pthread_mutex_trylock(mutexptr), 0);
        assert_eq!((*ptr.ptr).1, 3);
    }
    bomb.defuse();
}

fn check_rwlock_write() {
    let bomb = AbortOnDrop;
    unsafe {
        let data = SyncUnsafeCell::new((libc::PTHREAD_RWLOCK_INITIALIZER, 0));
        let ptr = SendPtr { ptr: data.get() };
        let mut threads = Vec::new();

        for _ in 0..3 {
            let thread = thread::spawn(move || {
                let ptr = ptr; // circumvent per-field closure capture
                let rwlockptr = ptr::addr_of_mut!((*ptr.ptr).0);
                assert_eq!(libc::pthread_rwlock_wrlock(rwlockptr), 0);
                thread::yield_now();
                (*ptr.ptr).1 += 1;
                assert_eq!(libc::pthread_rwlock_unlock(rwlockptr), 0);
            });
            threads.push(thread);

            let readthread = thread::spawn(move || {
                let ptr = ptr; // circumvent per-field closure capture
                let rwlockptr = ptr::addr_of_mut!((*ptr.ptr).0);
                assert_eq!(libc::pthread_rwlock_rdlock(rwlockptr), 0);
                thread::yield_now();
                let val = (*ptr.ptr).1;
                assert!(val >= 0 && val <= 3);
                assert_eq!(libc::pthread_rwlock_unlock(rwlockptr), 0);
            });
            threads.push(readthread);
        }

        for thread in threads {
            thread.join().unwrap();
        }

        let rwlockptr = ptr::addr_of_mut!((*ptr.ptr).0);
        assert_eq!(libc::pthread_rwlock_tryrdlock(rwlockptr), 0);
        assert_eq!((*ptr.ptr).1, 3);
    }
    bomb.defuse();
}

fn check_rwlock_read_no_deadlock() {
    let bomb = AbortOnDrop;
    unsafe {
        let l1 = SyncUnsafeCell::new(libc::PTHREAD_RWLOCK_INITIALIZER);
        let l1 = SendPtr { ptr: l1.get() };
        let l2 = SyncUnsafeCell::new(libc::PTHREAD_RWLOCK_INITIALIZER);
        let l2 = SendPtr { ptr: l2.get() };

        // acquire l1 and hold it until after the other thread is done
        assert_eq!(libc::pthread_rwlock_rdlock(l1.ptr), 0);
        let handle = thread::spawn(move || {
            let l1 = l1; // circumvent per-field closure capture
            let l2 = l2; // circumvent per-field closure capture
            // acquire l2 before the other thread
            assert_eq!(libc::pthread_rwlock_rdlock(l2.ptr), 0);
            thread::yield_now();
            assert_eq!(libc::pthread_rwlock_rdlock(l1.ptr), 0);
            thread::yield_now();
            assert_eq!(libc::pthread_rwlock_unlock(l1.ptr), 0);
            assert_eq!(libc::pthread_rwlock_unlock(l2.ptr), 0);
        });
        thread::yield_now();
        assert_eq!(libc::pthread_rwlock_rdlock(l2.ptr), 0);
        handle.join().unwrap();
    }
    bomb.defuse();
}

fn check_cond() {
    let bomb = AbortOnDrop;
    unsafe {
        let mut cond: MaybeUninit<libc::pthread_cond_t> = MaybeUninit::uninit();
        assert_eq!(libc::pthread_cond_init(cond.as_mut_ptr(), ptr::null()), 0);
        let cond = SendPtr { ptr: cond.as_mut_ptr() };

        let mut mutex: libc::pthread_mutex_t = libc::PTHREAD_MUTEX_INITIALIZER;
        let mutex = SendPtr { ptr: &mut mutex };

        let mut data = 0;
        let data = SendPtr { ptr: &mut data };

        let t = thread::spawn(move || {
            let mutex = mutex; // circumvent per-field closure capture
            let cond = cond;
            let data = data;
            assert_eq!(libc::pthread_mutex_lock(mutex.ptr), 0);
            assert!(data.ptr.read() == 0);
            data.ptr.write(1);
            libc::pthread_cond_wait(cond.ptr, mutex.ptr);
            assert!(data.ptr.read() == 3);
            data.ptr.write(4);
            assert_eq!(libc::pthread_mutex_unlock(mutex.ptr), 0);
        });

        thread::yield_now();

        assert_eq!(libc::pthread_mutex_lock(mutex.ptr), 0);
        assert!(data.ptr.read() == 1);
        data.ptr.write(2);
        assert_eq!(libc::pthread_cond_signal(cond.ptr), 0);
        thread::yield_now(); // the other thread wakes up but can't get the lock yet
        assert!(data.ptr.read() == 2);
        data.ptr.write(3);
        assert_eq!(libc::pthread_mutex_unlock(mutex.ptr), 0);

        thread::yield_now(); // now the other thread gets the lock back

        assert_eq!(libc::pthread_mutex_lock(mutex.ptr), 0);
        assert!(data.ptr.read() == 4);
        assert_eq!(libc::pthread_cond_broadcast(cond.ptr), 0); // just a smoke test
        assert_eq!(libc::pthread_mutex_unlock(mutex.ptr), 0);

        t.join().unwrap();
    }
    bomb.defuse();
}

fn check_condattr() {
    unsafe {
        // Just smoke-testing that these functions can be called.
        let mut attr: MaybeUninit<libc::pthread_condattr_t> = MaybeUninit::uninit();
        assert_eq!(libc::pthread_condattr_init(attr.as_mut_ptr()), 0);

        #[cfg(not(target_os = "macos"))] // setclock-getclock do not exist on macOS
        {
            let clock_id = libc::CLOCK_MONOTONIC;
            assert_eq!(libc::pthread_condattr_setclock(attr.as_mut_ptr(), clock_id), 0);
            let mut check_clock_id = MaybeUninit::<libc::clockid_t>::uninit();
            assert_eq!(
                libc::pthread_condattr_getclock(attr.as_mut_ptr(), check_clock_id.as_mut_ptr()),
                0
            );
            assert_eq!(check_clock_id.assume_init(), clock_id);
        }

        let mut cond: MaybeUninit<libc::pthread_cond_t> = MaybeUninit::uninit();
        assert_eq!(libc::pthread_cond_init(cond.as_mut_ptr(), attr.as_ptr()), 0);
        assert_eq!(libc::pthread_condattr_destroy(attr.as_mut_ptr()), 0);
        assert_eq!(libc::pthread_cond_destroy(cond.as_mut_ptr()), 0);
    }
}

// std::sync::RwLock does not even used pthread_rwlock any more.
// Do some smoke testing of the API surface.
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
