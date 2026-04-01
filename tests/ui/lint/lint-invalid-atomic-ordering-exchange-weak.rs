//@ only-x86_64
use std::sync::atomic::{AtomicPtr, Ordering};

fn main() {
    let ptr = &mut 5;
    let ptr2 = &mut 10;
    // `compare_exchange_weak` testing
    let x = AtomicPtr::new(ptr);

    // Allowed ordering combos
    let _ = x.compare_exchange_weak(ptr, ptr2, Ordering::Relaxed, Ordering::Relaxed);
    let _ = x.compare_exchange_weak(ptr, ptr2, Ordering::Relaxed, Ordering::Acquire);
    let _ = x.compare_exchange_weak(ptr, ptr2, Ordering::Relaxed, Ordering::SeqCst);
    let _ = x.compare_exchange_weak(ptr, ptr2, Ordering::Acquire, Ordering::Relaxed);
    let _ = x.compare_exchange_weak(ptr, ptr2, Ordering::Acquire, Ordering::Acquire);
    let _ = x.compare_exchange_weak(ptr2, ptr, Ordering::Acquire, Ordering::SeqCst);
    let _ = x.compare_exchange_weak(ptr, ptr2, Ordering::Release, Ordering::Relaxed);
    let _ = x.compare_exchange_weak(ptr2, ptr, Ordering::Release, Ordering::Acquire);
    let _ = x.compare_exchange_weak(ptr2, ptr, Ordering::Release, Ordering::SeqCst);
    let _ = x.compare_exchange_weak(ptr, ptr2, Ordering::AcqRel, Ordering::Relaxed);
    let _ = x.compare_exchange_weak(ptr, ptr2, Ordering::AcqRel, Ordering::Acquire);
    let _ = x.compare_exchange_weak(ptr2, ptr, Ordering::AcqRel, Ordering::SeqCst);
    let _ = x.compare_exchange_weak(ptr, ptr2, Ordering::SeqCst, Ordering::Relaxed);
    let _ = x.compare_exchange_weak(ptr, ptr2, Ordering::SeqCst, Ordering::Acquire);
    let _ = x.compare_exchange_weak(ptr, ptr2, Ordering::SeqCst, Ordering::SeqCst);

    // AcqRel is always forbidden as a failure ordering
    let _ = x.compare_exchange_weak(ptr2, ptr, Ordering::Relaxed, Ordering::AcqRel);
    //~^ ERROR `compare_exchange_weak`'s failure ordering may not be `Release` or `AcqRel`
    let _ = x.compare_exchange_weak(ptr2, ptr, Ordering::Acquire, Ordering::AcqRel);
    //~^ ERROR `compare_exchange_weak`'s failure ordering may not be `Release` or `AcqRel`
    let _ = x.compare_exchange_weak(ptr2, ptr, Ordering::Release, Ordering::AcqRel);
    //~^ ERROR `compare_exchange_weak`'s failure ordering may not be `Release` or `AcqRel`
    let _ = x.compare_exchange_weak(ptr2, ptr, Ordering::AcqRel, Ordering::AcqRel);
    //~^ ERROR `compare_exchange_weak`'s failure ordering may not be `Release` or `AcqRel`
    let _ = x.compare_exchange_weak(ptr2, ptr, Ordering::SeqCst, Ordering::AcqRel);
    //~^ ERROR `compare_exchange_weak`'s failure ordering may not be `Release` or `AcqRel`

    // Release is always forbidden as a failure ordering
    let _ = x.compare_exchange_weak(ptr, ptr2, Ordering::Relaxed, Ordering::Release);
    //~^ ERROR `compare_exchange_weak`'s failure ordering may not be `Release` or `AcqRel`
    let _ = x.compare_exchange_weak(ptr, ptr2, Ordering::Acquire, Ordering::Release);
    //~^ ERROR `compare_exchange_weak`'s failure ordering may not be `Release` or `AcqRel`
    let _ = x.compare_exchange_weak(ptr, ptr2, Ordering::Release, Ordering::Release);
    //~^ ERROR `compare_exchange_weak`'s failure ordering may not be `Release` or `AcqRel`
    let _ = x.compare_exchange_weak(ptr, ptr2, Ordering::AcqRel, Ordering::Release);
    //~^ ERROR `compare_exchange_weak`'s failure ordering may not be `Release` or `AcqRel`
    let _ = x.compare_exchange_weak(ptr, ptr2, Ordering::SeqCst, Ordering::Release);
    //~^ ERROR `compare_exchange_weak`'s failure ordering may not be `Release` or `AcqRel`
}
