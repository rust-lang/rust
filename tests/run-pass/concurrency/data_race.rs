// ignore-windows: Concurrency on Windows is not supported yet.


use std::sync::atomic::{AtomicUsize, fence, Ordering};
use std::thread::spawn;

#[derive(Copy, Clone)]
struct EvilSend<T>(pub T);

unsafe impl<T> Send for EvilSend<T> {}
unsafe impl<T> Sync for EvilSend<T> {}

static SYNC: AtomicUsize = AtomicUsize::new(0);

fn test_fence_sync() {
    let mut var = 0u32;
    let ptr = &mut var as *mut u32;
    let evil_ptr = EvilSend(ptr);
    
    
    let j1 = spawn(move || {
        unsafe { *evil_ptr.0 = 1; }
        fence(Ordering::Release);
        SYNC.store(1, Ordering::Relaxed)   
    });

    let j2 = spawn(move || {
        if SYNC.load(Ordering::Relaxed) == 1 {
            fence(Ordering::Acquire);
            unsafe { *evil_ptr.0 }
        } else {
            0
        }
    });

    j1.join().unwrap();
    j2.join().unwrap();
}


fn test_multiple_reads() {
    let mut var = 42u32;
    let ptr = &mut var as *mut u32;
    let evil_ptr = EvilSend(ptr);

    let j1 = spawn(move || unsafe {*evil_ptr.0});
    let j2 = spawn(move || unsafe {*evil_ptr.0});
    let j3 = spawn(move || unsafe {*evil_ptr.0});
    let j4 = spawn(move || unsafe {*evil_ptr.0});

    assert_eq!(j1.join().unwrap(), 42);
    assert_eq!(j2.join().unwrap(), 42);
    assert_eq!(j3.join().unwrap(), 42);
    assert_eq!(j4.join().unwrap(), 42);

    var = 10;
    assert_eq!(var, 10);
}

pub fn test_rmw_no_block() {
    let mut a = 0u32;
    let b = &mut a as *mut u32;
    let c = EvilSend(b);

    unsafe {
        let j1 = spawn(move || {
            *c.0 = 1;
            SYNC.store(1, Ordering::Release);
        });

        let j2 = spawn(move || {
            if SYNC.swap(2, Ordering::Relaxed) == 1 {
                //No op, blocking store removed
            }
        });

        let j3 = spawn(move || {
            if SYNC.load(Ordering::Acquire) == 2 {
                *c.0
            } else {
                0
            }
        });

        j1.join().unwrap();
        j2.join().unwrap();
        let v = j3.join().unwrap();
        assert!(v == 1 || v == 2);
    }
}

pub fn test_simple_release() {
    let mut a = 0u32;
    let b = &mut a as *mut u32;
    let c = EvilSend(b);

    unsafe {
        let j1 = spawn(move || {
            *c.0 = 1;
            SYNC.store(1, Ordering::Release);
        });

        let j2 = spawn(move || {
            if SYNC.load(Ordering::Acquire) == 1 {
                *c.0
            } else {
                0
            }
        });

        j1.join().unwrap();
        assert_eq!(j2.join().unwrap(),1);
    }
}

pub fn main() {
    test_fence_sync();
    test_multiple_reads();
    test_rmw_no_block();
    test_simple_release();
}
