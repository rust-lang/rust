//@ only-x86_64
use std::sync::atomic::{AtomicUsize, Ordering};

fn main() {
    // `compare_exchange` (not weak) testing
    let x = AtomicUsize::new(0);

    // Allowed ordering combos
    let _ = x.compare_exchange(0, 0, Ordering::Relaxed, Ordering::Relaxed);
    let _ = x.compare_exchange(0, 0, Ordering::Relaxed, Ordering::Acquire);
    let _ = x.compare_exchange(0, 0, Ordering::Relaxed, Ordering::SeqCst);
    let _ = x.compare_exchange(0, 0, Ordering::Acquire, Ordering::Relaxed);
    let _ = x.compare_exchange(0, 0, Ordering::Acquire, Ordering::Acquire);
    let _ = x.compare_exchange(0, 0, Ordering::Acquire, Ordering::SeqCst);
    let _ = x.compare_exchange(0, 0, Ordering::Release, Ordering::Relaxed);
    let _ = x.compare_exchange(0, 0, Ordering::Release, Ordering::Acquire);
    let _ = x.compare_exchange(0, 0, Ordering::Release, Ordering::SeqCst);
    let _ = x.compare_exchange(0, 0, Ordering::AcqRel, Ordering::Relaxed);
    let _ = x.compare_exchange(0, 0, Ordering::AcqRel, Ordering::Acquire);
    let _ = x.compare_exchange(0, 0, Ordering::AcqRel, Ordering::SeqCst);
    let _ = x.compare_exchange(0, 0, Ordering::SeqCst, Ordering::Relaxed);
    let _ = x.compare_exchange(0, 0, Ordering::SeqCst, Ordering::Acquire);
    let _ = x.compare_exchange(0, 0, Ordering::SeqCst, Ordering::SeqCst);

    // AcqRel is always forbidden as a failure ordering
    let _ = x.compare_exchange(0, 0, Ordering::Relaxed, Ordering::AcqRel);
    //~^ ERROR `compare_exchange`'s failure ordering may not be `Release` or `AcqRel`
    let _ = x.compare_exchange(0, 0, Ordering::Acquire, Ordering::AcqRel);
    //~^ ERROR `compare_exchange`'s failure ordering may not be `Release` or `AcqRel`
    let _ = x.compare_exchange(0, 0, Ordering::Release, Ordering::AcqRel);
    //~^ ERROR `compare_exchange`'s failure ordering may not be `Release` or `AcqRel`
    let _ = x.compare_exchange(0, 0, Ordering::AcqRel, Ordering::AcqRel);
    //~^ ERROR `compare_exchange`'s failure ordering may not be `Release` or `AcqRel`
    let _ = x.compare_exchange(0, 0, Ordering::SeqCst, Ordering::AcqRel);
    //~^ ERROR `compare_exchange`'s failure ordering may not be `Release` or `AcqRel`

    // Release is always forbidden as a failure ordering
    let _ = x.compare_exchange(0, 0, Ordering::Relaxed, Ordering::Release);
    //~^ ERROR `compare_exchange`'s failure ordering may not be `Release` or `AcqRel`
    let _ = x.compare_exchange(0, 0, Ordering::Acquire, Ordering::Release);
    //~^ ERROR `compare_exchange`'s failure ordering may not be `Release` or `AcqRel`
    let _ = x.compare_exchange(0, 0, Ordering::Release, Ordering::Release);
    //~^ ERROR `compare_exchange`'s failure ordering may not be `Release` or `AcqRel`
    let _ = x.compare_exchange(0, 0, Ordering::AcqRel, Ordering::Release);
    //~^ ERROR `compare_exchange`'s failure ordering may not be `Release` or `AcqRel`
    let _ = x.compare_exchange(0, 0, Ordering::SeqCst, Ordering::Release);
    //~^ ERROR `compare_exchange`'s failure ordering may not be `Release` or `AcqRel`
}
