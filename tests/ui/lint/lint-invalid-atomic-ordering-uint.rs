//@ only-x86_64
use std::sync::atomic::{AtomicU16, AtomicU32, AtomicU64, AtomicU8, AtomicUsize, Ordering};

fn main() {
    // `AtomicU8` test cases
    let x = AtomicU8::new(0);

    // Allowed load ordering modes
    let _ = x.load(Ordering::Acquire);
    let _ = x.load(Ordering::SeqCst);
    let _ = x.load(Ordering::Relaxed);

    // Allowed store ordering modes
    x.store(1, Ordering::Release);
    x.store(1, Ordering::SeqCst);
    x.store(1, Ordering::Relaxed);

    // Disallowed load ordering modes
    let _ = x.load(Ordering::Release);
    //~^ ERROR atomic loads cannot have `Release` or `AcqRel` ordering
    let _ = x.load(Ordering::AcqRel);
    //~^ ERROR atomic loads cannot have `Release` or `AcqRel` ordering

    // Disallowed store ordering modes
    x.store(1, Ordering::Acquire);
    //~^ ERROR atomic stores cannot have `Acquire` or `AcqRel` ordering
    x.store(1, Ordering::AcqRel);
    //~^ ERROR atomic stores cannot have `Acquire` or `AcqRel` ordering

    // `AtomicU16` test cases
    let x = AtomicU16::new(0);

    // Allowed load ordering modes
    let _ = x.load(Ordering::Acquire);
    let _ = x.load(Ordering::SeqCst);
    let _ = x.load(Ordering::Relaxed);

    // Allowed store ordering modes
    x.store(1, Ordering::Release);
    x.store(1, Ordering::SeqCst);
    x.store(1, Ordering::Relaxed);

    // Disallowed load ordering modes
    let _ = x.load(Ordering::Release);
    //~^ ERROR atomic loads cannot have `Release` or `AcqRel` ordering
    let _ = x.load(Ordering::AcqRel);
    //~^ ERROR atomic loads cannot have `Release` or `AcqRel` ordering

    // Disallowed store ordering modes
    x.store(1, Ordering::Acquire);
    //~^ ERROR atomic stores cannot have `Acquire` or `AcqRel` ordering
    x.store(1, Ordering::AcqRel);
    //~^ ERROR atomic stores cannot have `Acquire` or `AcqRel` ordering

    // `AtomicU32` test cases
    let x = AtomicU32::new(0);

    // Allowed load ordering modes
    let _ = x.load(Ordering::Acquire);
    let _ = x.load(Ordering::SeqCst);
    let _ = x.load(Ordering::Relaxed);

    // Allowed store ordering modes
    x.store(1, Ordering::Release);
    x.store(1, Ordering::SeqCst);
    x.store(1, Ordering::Relaxed);

    // Disallowed load ordering modes
    let _ = x.load(Ordering::Release);
    //~^ ERROR atomic loads cannot have `Release` or `AcqRel` ordering
    let _ = x.load(Ordering::AcqRel);
    //~^ ERROR atomic loads cannot have `Release` or `AcqRel` ordering

    // Disallowed store ordering modes
    x.store(1, Ordering::Acquire);
    //~^ ERROR atomic stores cannot have `Acquire` or `AcqRel` ordering
    x.store(1, Ordering::AcqRel);
    //~^ ERROR atomic stores cannot have `Acquire` or `AcqRel` ordering

    // `AtomicU64` test cases
    let x = AtomicU64::new(0);

    // Allowed load ordering modes
    let _ = x.load(Ordering::Acquire);
    let _ = x.load(Ordering::SeqCst);
    let _ = x.load(Ordering::Relaxed);

    // Allowed store ordering modes
    x.store(1, Ordering::Release);
    x.store(1, Ordering::SeqCst);
    x.store(1, Ordering::Relaxed);

    // Disallowed load ordering modes
    let _ = x.load(Ordering::Release);
    //~^ ERROR atomic loads cannot have `Release` or `AcqRel` ordering
    let _ = x.load(Ordering::AcqRel);
    //~^ ERROR atomic loads cannot have `Release` or `AcqRel` ordering

    // Disallowed store ordering modes
    x.store(1, Ordering::Acquire);
    //~^ ERROR atomic stores cannot have `Acquire` or `AcqRel` ordering
    x.store(1, Ordering::AcqRel);
    //~^ ERROR atomic stores cannot have `Acquire` or `AcqRel` ordering

    // `AtomicUsize` test cases
    let x = AtomicUsize::new(0);

    // Allowed load ordering modes
    let _ = x.load(Ordering::Acquire);
    let _ = x.load(Ordering::SeqCst);
    let _ = x.load(Ordering::Relaxed);

    // Allowed store ordering modes
    x.store(1, Ordering::Release);
    x.store(1, Ordering::SeqCst);
    x.store(1, Ordering::Relaxed);

    // Disallowed load ordering modes
    let _ = x.load(Ordering::Release);
    //~^ ERROR atomic loads cannot have `Release` or `AcqRel` ordering
    let _ = x.load(Ordering::AcqRel);
    //~^ ERROR atomic loads cannot have `Release` or `AcqRel` ordering

    // Disallowed store ordering modes
    x.store(1, Ordering::Acquire);
    //~^ ERROR atomic stores cannot have `Acquire` or `AcqRel` ordering
    x.store(1, Ordering::AcqRel);
    //~^ ERROR atomic stores cannot have `Acquire` or `AcqRel` ordering
}
