//@only-target-windows: Uses win32 api functions
// We are making scheduler assumptions here.
//@compile-flags: -Zmiri-preemption-rate=0

use std::ffi::c_void;
use std::ptr::null_mut;
use std::thread;

#[derive(Copy, Clone)]
struct SendPtr<T>(*mut T);

unsafe impl<T> Send for SendPtr<T> {}

extern "system" {
    fn SleepConditionVariableSRW(
        condvar: *mut *mut c_void,
        lock: *mut *mut c_void,
        timeout: u32,
        flags: u32,
    ) -> i32;
    fn WakeAllConditionVariable(condvar: *mut *mut c_void);

    fn AcquireSRWLockExclusive(lock: *mut *mut c_void);
    fn AcquireSRWLockShared(lock: *mut *mut c_void);
    fn ReleaseSRWLockExclusive(lock: *mut *mut c_void);
    fn ReleaseSRWLockShared(lock: *mut *mut c_void);
}

const CONDITION_VARIABLE_LOCKMODE_SHARED: u32 = 1;
const INFINITE: u32 = u32::MAX;

/// threads should be able to reacquire the lock while it is locked by multiple other threads in shared mode
fn all_shared() {
    println!("all_shared");

    let mut lock = null_mut();
    let mut condvar = null_mut();

    let lock_ptr = SendPtr(&mut lock);
    let condvar_ptr = SendPtr(&mut condvar);

    let mut handles = Vec::with_capacity(10);

    // waiters
    for i in 0..5 {
        handles.push(thread::spawn(move || {
            let condvar_ptr = condvar_ptr; // avoid field capture
            let lock_ptr = lock_ptr; // avoid field capture
            unsafe {
                AcquireSRWLockShared(lock_ptr.0);
            }
            println!("exclusive waiter {i} locked");

            let r = unsafe {
                SleepConditionVariableSRW(
                    condvar_ptr.0,
                    lock_ptr.0,
                    INFINITE,
                    CONDITION_VARIABLE_LOCKMODE_SHARED,
                )
            };
            assert_ne!(r, 0);

            println!("exclusive waiter {i} reacquired lock");

            // unlocking is unnecessary because the lock is never used again
        }));
    }

    // ensures each waiter is waiting by this point
    thread::yield_now();

    // readers
    for i in 0..5 {
        handles.push(thread::spawn(move || {
            let lock_ptr = lock_ptr; // avoid field capture
            unsafe {
                AcquireSRWLockShared(lock_ptr.0);
            }
            println!("reader {i} locked");

            // switch to next reader or main thread
            thread::yield_now();

            unsafe {
                ReleaseSRWLockShared(lock_ptr.0);
            }
            println!("reader {i} unlocked");
        }));
    }

    // ensures each reader has acquired the lock
    thread::yield_now();

    unsafe {
        WakeAllConditionVariable(condvar_ptr.0);
    }

    for handle in handles {
        handle.join().unwrap();
    }
}

// reacquiring a lock should wait until the lock is not exclusively locked
fn shared_sleep_and_exclusive_lock() {
    println!("shared_sleep_and_exclusive_lock");

    let mut lock = null_mut();
    let mut condvar = null_mut();

    let lock_ptr = SendPtr(&mut lock);
    let condvar_ptr = SendPtr(&mut condvar);

    let mut waiters = Vec::with_capacity(5);
    for i in 0..5 {
        waiters.push(thread::spawn(move || {
            let lock_ptr = lock_ptr; // avoid field capture
            let condvar_ptr = condvar_ptr; // avoid field capture
            unsafe {
                AcquireSRWLockShared(lock_ptr.0);
            }
            println!("shared waiter {i} locked");

            let r = unsafe {
                SleepConditionVariableSRW(
                    condvar_ptr.0,
                    lock_ptr.0,
                    INFINITE,
                    CONDITION_VARIABLE_LOCKMODE_SHARED,
                )
            };
            assert_ne!(r, 0);

            println!("shared waiter {i} reacquired lock");

            // unlocking is unnecessary because the lock is never used again
        }));
    }

    // ensures each waiter is waiting by this point
    thread::yield_now();

    unsafe {
        AcquireSRWLockExclusive(lock_ptr.0);
    }
    println!("main locked");

    unsafe {
        WakeAllConditionVariable(condvar_ptr.0);
    }

    // waiters are now waiting for the lock to be unlocked
    thread::yield_now();

    unsafe {
        ReleaseSRWLockExclusive(lock_ptr.0);
    }
    println!("main unlocked");

    for handle in waiters {
        handle.join().unwrap();
    }
}

// threads reacquiring locks should wait for all locks to be released first
fn exclusive_sleep_and_shared_lock() {
    println!("exclusive_sleep_and_shared_lock");

    let mut lock = null_mut();
    let mut condvar = null_mut();

    let lock_ptr = SendPtr(&mut lock);
    let condvar_ptr = SendPtr(&mut condvar);

    let mut handles = Vec::with_capacity(10);
    for i in 0..5 {
        handles.push(thread::spawn(move || {
            let lock_ptr = lock_ptr; // avoid field capture
            let condvar_ptr = condvar_ptr; // avoid field capture
            unsafe {
                AcquireSRWLockExclusive(lock_ptr.0);
            }

            println!("exclusive waiter {i} locked");

            let r = unsafe { SleepConditionVariableSRW(condvar_ptr.0, lock_ptr.0, INFINITE, 0) };
            assert_ne!(r, 0);

            println!("exclusive waiter {i} reacquired lock");

            // switch to next waiter or main thread
            thread::yield_now();

            unsafe {
                ReleaseSRWLockExclusive(lock_ptr.0);
            }
            println!("exclusive waiter {i} unlocked");
        }));
    }

    for i in 0..5 {
        handles.push(thread::spawn(move || {
            let lock_ptr = lock_ptr; // avoid field capture
            unsafe {
                AcquireSRWLockShared(lock_ptr.0);
            }
            println!("reader {i} locked");

            // switch to next reader or main thread
            thread::yield_now();

            unsafe {
                ReleaseSRWLockShared(lock_ptr.0);
            }
            println!("reader {i} unlocked");
        }));
    }

    // ensures each reader has acquired the lock
    thread::yield_now();

    unsafe {
        WakeAllConditionVariable(condvar_ptr.0);
    }

    for handle in handles {
        handle.join().unwrap();
    }
}

fn main() {
    all_shared();
    shared_sleep_and_exclusive_lock();
    exclusive_sleep_and_shared_lock();
}
