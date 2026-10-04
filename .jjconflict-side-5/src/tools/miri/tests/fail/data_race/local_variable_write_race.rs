//@compile-flags:-Zmiri-deterministic-concurrency
use std::sync::atomic::Ordering::*;
use std::sync::atomic::*;

static P: AtomicPtr<u8> = AtomicPtr::new(core::ptr::null_mut());

fn main() {
    let t1 = std::thread::spawn(|| {
        while P.load(Relaxed).is_null() {
            std::hint::spin_loop();
        }
        unsafe {
            // Initialize `*P`.
            let ptr = P.load(Relaxed);
            *ptr = 127;
            //~^ ERROR: Data race detected between (1) non-atomic write on thread `main` and (2) non-atomic write on thread `unnamed-1`
        }
    });

    // Create the local variable, and initialize it.
    // This is not ordered with the store above, so it's definitely UB
    // for that thread to access this variable.
    let mut val: u8 = 0;

    // Actually generate memory for the local variable.
    // This is the time its value is actually written to memory.
    // If we just "pre-date" the write to the beginning of time (since we don't know
    // when it actually happened), we'd miss the UB in this test.
    // Also, the UB error should point at the write above, not the addr-of here.
    P.store(std::ptr::addr_of_mut!(val), Relaxed);

    // Wait for the thread to be done.
    t1.join().unwrap();

    // Read initialized value.
    assert_eq!(val, 127);
}
