//@ only-x86_64
use std::sync::atomic::{AtomicPtr, Ordering};

fn main() {
    let ptr = &mut 5;
    let other_ptr = &mut 10;
    let x = AtomicPtr::new(ptr);

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
    x.store(other_ptr, Ordering::Release);
    x.store(other_ptr, Ordering::SeqCst);
    x.store(other_ptr, Ordering::Relaxed);

    // Disallowed store ordering modes
    x.store(other_ptr, Ordering::Acquire);
    //~^ ERROR atomic stores cannot have `Acquire` or `AcqRel` ordering
    x.store(other_ptr, Ordering::AcqRel);
    //~^ ERROR atomic stores cannot have `Acquire` or `AcqRel` ordering
}
