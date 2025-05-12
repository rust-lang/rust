// This tests carefully crafted schedules to ensure they are not considered races.
//@compile-flags: -Zmiri-deterministic-concurrency

use std::sync::atomic::*;
use std::thread::{self, spawn};

#[derive(Copy, Clone)]
struct EvilSend<T>(pub T);

unsafe impl<T> Send for EvilSend<T> {}
unsafe impl<T> Sync for EvilSend<T> {}

fn test_fence_sync() {
    static SYNC: AtomicUsize = AtomicUsize::new(0);

    let mut var = 0u32;
    let ptr = &mut var as *mut u32;
    let evil_ptr = EvilSend(ptr);

    let j1 = spawn(move || {
        let evil_ptr = evil_ptr; // avoid field capturing
        unsafe { *evil_ptr.0 = 1 };
        fence(Ordering::Release);
        SYNC.store(1, Ordering::Relaxed)
    });

    let j2 = spawn(move || {
        let evil_ptr = evil_ptr; // avoid field capturing
        if SYNC.load(Ordering::Relaxed) == 1 {
            fence(Ordering::Acquire);
            unsafe { *evil_ptr.0 }
        } else {
            panic!(); // relies on thread 2 going last
        }
    });

    j1.join().unwrap();
    j2.join().unwrap();
}

fn test_multiple_reads() {
    let mut var = 42u32;
    let ptr = &mut var as *mut u32;
    let evil_ptr = EvilSend(ptr);

    let j1 = spawn(move || unsafe { *{ evil_ptr }.0 });
    let j2 = spawn(move || unsafe { *{ evil_ptr }.0 });
    let j3 = spawn(move || unsafe { *{ evil_ptr }.0 });
    let j4 = spawn(move || unsafe { *{ evil_ptr }.0 });

    assert_eq!(j1.join().unwrap(), 42);
    assert_eq!(j2.join().unwrap(), 42);
    assert_eq!(j3.join().unwrap(), 42);
    assert_eq!(j4.join().unwrap(), 42);

    var = 10;
    assert_eq!(var, 10);
}

pub fn test_rmw_no_block() {
    static SYNC: AtomicUsize = AtomicUsize::new(0);

    let mut a = 0u32;
    let b = &mut a as *mut u32;
    let c = EvilSend(b);

    unsafe {
        let j1 = spawn(move || {
            let c = c; // avoid field capturing
            *c.0 = 1;
            SYNC.store(1, Ordering::Release);
        });

        let j2 = spawn(move || {
            if SYNC.swap(2, Ordering::Relaxed) == 1 {
                //No op, blocking store removed
            }
        });

        let j3 = spawn(move || {
            let c = c; // avoid field capturing
            if SYNC.load(Ordering::Acquire) == 2 { *c.0 } else { 0 }
        });

        j1.join().unwrap();
        j2.join().unwrap();
        let v = j3.join().unwrap();
        assert!(v == 1 || v == 2); // relies on thread 3 going last
    }
}

pub fn test_simple_release() {
    static SYNC: AtomicUsize = AtomicUsize::new(0);

    let mut a = 0u32;
    let b = &mut a as *mut u32;
    let c = EvilSend(b);

    unsafe {
        let j1 = spawn(move || {
            let c = c; // avoid field capturing
            *c.0 = 1;
            SYNC.store(1, Ordering::Release);
        });

        let j2 = spawn(move || {
            let c = c; // avoid field capturing
            if SYNC.load(Ordering::Acquire) == 1 { *c.0 } else { 0 }
        });

        j1.join().unwrap();
        assert_eq!(j2.join().unwrap(), 1); // relies on thread 2 going last
    }
}

fn test_local_variable_lazy_write() {
    static P: AtomicPtr<u8> = AtomicPtr::new(core::ptr::null_mut());

    // Create the local variable, and initialize it.
    // This write happens before the thread is spanwed, so there is no data race.
    let mut val: u8 = 0;

    let t1 = std::thread::spawn(|| {
        while P.load(Ordering::Relaxed).is_null() {
            std::hint::spin_loop();
        }
        unsafe {
            // Initialize `*P`.
            let ptr = P.load(Ordering::Relaxed);
            *ptr = 127;
        }
    });

    // Actually generate memory for the local variable.
    // This is the time its value is actually written to memory:
    // that's *after* the thread above was spawned!
    // This may hence look like a data race wrt the access in the thread above.
    P.store(std::ptr::addr_of_mut!(val), Ordering::Relaxed);

    // Wait for the thread to be done.
    t1.join().unwrap();

    // Read initialized value.
    assert_eq!(val, 127);
}

// This test coverse the case where the non-atomic access come first.
fn test_read_read_race1() {
    let a = AtomicU16::new(0);

    thread::scope(|s| {
        s.spawn(|| {
            let ptr = &a as *const AtomicU16 as *mut u16;
            unsafe { ptr.read() };
        });
        s.spawn(|| {
            thread::yield_now();

            a.load(Ordering::SeqCst);
        });
    });
}

// This test coverse the case where the atomic access come first.
fn test_read_read_race2() {
    let a = AtomicU16::new(0);

    thread::scope(|s| {
        s.spawn(|| {
            a.load(Ordering::SeqCst);
        });
        s.spawn(|| {
            thread::yield_now();

            let ptr = &a as *const AtomicU16 as *mut u16;
            unsafe { ptr.read() };
        });
    });
}

fn mixed_size_read_read() {
    fn convert(a: &AtomicU16) -> &[AtomicU8; 2] {
        unsafe { std::mem::transmute(a) }
    }

    let a = AtomicU16::new(0);
    let a16 = &a;
    let a8 = convert(a16);

    // Just two different-sized atomic reads without any writes are fine.
    thread::scope(|s| {
        s.spawn(|| {
            a16.load(Ordering::SeqCst);
        });
        s.spawn(|| {
            a8[0].load(Ordering::SeqCst);
        });
    });
}

fn failing_rmw_is_read() {
    let a = AtomicUsize::new(0);
    thread::scope(|s| {
        s.spawn(|| unsafe {
            // Non-atomic read.
            let _val = *(&a as *const AtomicUsize).cast::<usize>();
        });

        s.spawn(|| {
            // RMW that will fail.
            // This is not considered a write, so there is no data race here.
            a.compare_exchange(1, 2, Ordering::SeqCst, Ordering::SeqCst).unwrap_err();
        });
    });
}

pub fn main() {
    test_fence_sync();
    test_multiple_reads();
    test_rmw_no_block();
    test_simple_release();
    test_local_variable_lazy_write();
    test_read_read_race1();
    test_read_read_race2();
    mixed_size_read_read();
    failing_rmw_is_read();
}
