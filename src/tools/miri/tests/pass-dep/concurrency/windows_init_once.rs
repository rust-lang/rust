//@only-target: windows # Uses win32 api functions
// We are making scheduler assumptions here.
//@compile-flags: -Zmiri-deterministic-concurrency

use std::ptr::null_mut;
use std::thread;

use windows_sys::Win32::Foundation::{FALSE, TRUE};
use windows_sys::Win32::System::Threading::{
    INIT_ONCE, INIT_ONCE_INIT_FAILED, InitOnceBeginInitialize, InitOnceComplete,
};

// not in windows-sys
const INIT_ONCE_STATIC_INIT: INIT_ONCE = INIT_ONCE { Ptr: null_mut() };

#[derive(Copy, Clone)]
struct SendPtr<T>(*mut T);

unsafe impl<T> Send for SendPtr<T> {}

fn single_thread() {
    let mut init_once = INIT_ONCE_STATIC_INIT;
    let mut pending = 0;

    unsafe {
        assert_eq!(InitOnceBeginInitialize(&mut init_once, 0, &mut pending, null_mut()), TRUE);
        assert_eq!(pending, TRUE);

        assert_eq!(InitOnceComplete(&mut init_once, 0, null_mut()), TRUE);

        assert_eq!(InitOnceBeginInitialize(&mut init_once, 0, &mut pending, null_mut()), TRUE);
        assert_eq!(pending, FALSE);
    }

    let mut init_once = INIT_ONCE_STATIC_INIT;

    unsafe {
        assert_eq!(InitOnceBeginInitialize(&mut init_once, 0, &mut pending, null_mut()), TRUE);
        assert_eq!(pending, TRUE);

        assert_eq!(InitOnceComplete(&mut init_once, INIT_ONCE_INIT_FAILED, null_mut()), TRUE);

        assert_eq!(InitOnceBeginInitialize(&mut init_once, 0, &mut pending, null_mut()), TRUE);
        assert_eq!(pending, TRUE);
    }
}

fn block_until_complete() {
    let mut init_once = INIT_ONCE_STATIC_INIT;
    let mut pending = 0;

    unsafe {
        assert_eq!(InitOnceBeginInitialize(&mut init_once, 0, &mut pending, null_mut()), TRUE);
        assert_eq!(pending, TRUE);
    }

    let init_once_ptr = SendPtr(&mut init_once);

    let waiter = move || unsafe {
        let init_once_ptr = init_once_ptr; // avoid field capture
        let mut pending = 0;

        assert_eq!(InitOnceBeginInitialize(init_once_ptr.0, 0, &mut pending, null_mut()), TRUE);
        assert_eq!(pending, FALSE);

        println!("finished waiting for initialization");
    };

    let waiter1 = thread::spawn(waiter);
    let waiter2 = thread::spawn(waiter);

    // this yield ensures `waiter1` & `waiter2` are blocked on the main thread
    thread::yield_now();

    println!("completing initialization");

    unsafe {
        assert_eq!(InitOnceComplete(init_once_ptr.0, 0, null_mut()), TRUE);
    }

    waiter1.join().unwrap();
    waiter2.join().unwrap();
}

fn retry_on_fail() {
    let mut init_once = INIT_ONCE_STATIC_INIT;
    let mut pending = 0;

    unsafe {
        assert_eq!(InitOnceBeginInitialize(&mut init_once, 0, &mut pending, null_mut()), TRUE);
        assert_eq!(pending, TRUE);
    }

    let init_once_ptr = SendPtr(&mut init_once);

    let waiter = move || unsafe {
        let init_once_ptr = init_once_ptr; // avoid field capture
        let mut pending = 0;

        assert_eq!(InitOnceBeginInitialize(init_once_ptr.0, 0, &mut pending, null_mut()), TRUE);

        if pending == 1 {
            println!("retrying initialization");

            assert_eq!(InitOnceComplete(init_once_ptr.0, 0, null_mut()), TRUE);
        } else {
            println!("finished waiting for initialization");
        }
    };

    let waiter1 = thread::spawn(waiter);
    let waiter2 = thread::spawn(waiter);

    // this yield ensures `waiter1` & `waiter2` are blocked on the main thread
    thread::yield_now();

    println!("failing initialization");

    unsafe {
        assert_eq!(InitOnceComplete(init_once_ptr.0, INIT_ONCE_INIT_FAILED, null_mut()), TRUE);
    }

    waiter1.join().unwrap();
    waiter2.join().unwrap();
}

fn no_data_race_after_complete() {
    let mut init_once = INIT_ONCE_STATIC_INIT;
    let mut pending = 0;

    unsafe {
        assert_eq!(InitOnceBeginInitialize(&mut init_once, 0, &mut pending, null_mut()), TRUE);
        assert_eq!(pending, TRUE);
    }

    let init_once_ptr = SendPtr(&mut init_once);

    let mut place = 0;
    let place_ptr = SendPtr(&mut place);

    let reader = thread::spawn(move || unsafe {
        let init_once_ptr = init_once_ptr; // avoid field capture
        let place_ptr = place_ptr; // avoid field capture
        let mut pending = 0;

        // this doesn't block because reader only executes after `InitOnceComplete` is called
        assert_eq!(InitOnceBeginInitialize(init_once_ptr.0, 0, &mut pending, null_mut()), TRUE);
        assert_eq!(pending, FALSE);
        // this should not data race
        place_ptr.0.read()
    });

    unsafe {
        // this should not data race
        place_ptr.0.write(1);
    }

    unsafe {
        assert_eq!(InitOnceComplete(init_once_ptr.0, 0, null_mut()), TRUE);
    }

    // run reader (without preemption, it has not taken a step yet)
    assert_eq!(reader.join().unwrap(), 1);
}

fn main() {
    single_thread();
    block_until_complete();
    retry_on_fail();
    no_data_race_after_complete();
}
