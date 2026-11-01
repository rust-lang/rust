//@ only-x86_64
use std::sync::atomic::{AtomicBool, Ordering};

fn main() {
    let x = AtomicBool::new(true);

    // Allowed load ordering modes
    let _ = x.load(Ordering::Acquire);
    let _ = x.load(Ordering::SeqCst);
    let _ = x.load(Ordering::Relaxed);

    // Disallowed load ordering modes
    let _ = x.load(Ordering::Release);
    //~^ ERROR atomic loads cannot have `Release` or `AcqRel` ordering
    let _ = x.load(Ordering::AcqRel);
    //~^ ERROR atomic loads cannot have `Release` or `AcqRel` ordering

    // Allowed store ordering modes
    x.store(false, Ordering::Release);
    x.store(false, Ordering::SeqCst);
    x.store(false, Ordering::Relaxed);

    // Disallowed store ordering modes
    x.store(false, Ordering::Acquire);
    //~^ ERROR atomic stores cannot have `Acquire` or `AcqRel` ordering
    x.store(false, Ordering::AcqRel);
    //~^ ERROR atomic stores cannot have `Acquire` or `AcqRel` ordering
}
