//@compile-flags:-Zmiri-deterministic-concurrency
#![feature(core_intrinsics)]
#![feature(custom_mir)]

use std::intrinsics::mir::*;
use std::sync::atomic::Ordering::*;
use std::sync::atomic::*;
use std::thread::JoinHandle;

static P: AtomicPtr<u8> = AtomicPtr::new(core::ptr::null_mut());

fn spawn_thread() -> JoinHandle<()> {
    std::thread::spawn(|| {
        while P.load(Relaxed).is_null() {
            std::hint::spin_loop();
        }
        unsafe {
            // Initialize `*P`.
            let ptr = P.load(Relaxed);
            *ptr = 127;
            //~^ ERROR: Data race detected between (1) creating a new allocation on thread `main` and (2) non-atomic write on thread `unnamed-1`
        }
    })
}

fn finish(t: JoinHandle<()>, val_ptr: *mut u8) {
    P.store(val_ptr, Relaxed);

    // Wait for the thread to be done.
    t.join().unwrap();

    // Read initialized value.
    assert_eq!(unsafe { *val_ptr }, 127);
}

#[custom_mir(dialect = "runtime", phase = "optimized")]
fn main() {
    mir! {
        let t;
        let val;
        let val_ptr;
        let _ret;
        {
            Call(t = spawn_thread(), ReturnTo(after_spawn), UnwindContinue())
        }
        after_spawn = {
            // This races with the write in the other thread.
            StorageLive(val);

            val_ptr = &raw mut val;
            Call(_ret = finish(t, val_ptr), ReturnTo(done), UnwindContinue())
        }
        done = {
            Return()
        }
    }
}
