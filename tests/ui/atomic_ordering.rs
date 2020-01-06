#![warn(clippy::invalid_atomic_ordering)]

use std::sync::atomic::{
    AtomicBool, AtomicI16, AtomicI32, AtomicI64, AtomicI8, AtomicIsize, AtomicPtr, AtomicU16, AtomicU32, AtomicU64,
    AtomicU8, AtomicUsize, Ordering,
};

fn main() {
    // `AtomicBool` test cases
    let x = AtomicBool::new(true);

    // Allowed load ordering modes
    let _ = x.load(Ordering::Acquire);
    let _ = x.load(Ordering::SeqCst);
    let _ = x.load(Ordering::Relaxed);

    // Disallowed load ordering modes
    let _ = x.load(Ordering::Release);
    let _ = x.load(Ordering::AcqRel);

    // Allowed store ordering modes
    x.store(false, Ordering::Release);
    x.store(false, Ordering::SeqCst);
    x.store(false, Ordering::Relaxed);

    // Disallowed store ordering modes
    x.store(false, Ordering::Acquire);
    x.store(false, Ordering::AcqRel);

    // `AtomicI8` test cases
    let x = AtomicI8::new(0);

    let _ = x.load(Ordering::Acquire);
    let _ = x.load(Ordering::SeqCst);
    let _ = x.load(Ordering::Relaxed);
    let _ = x.load(Ordering::Release);
    let _ = x.load(Ordering::AcqRel);

    x.store(1, Ordering::Release);
    x.store(1, Ordering::SeqCst);
    x.store(1, Ordering::Relaxed);
    x.store(1, Ordering::Acquire);
    x.store(1, Ordering::AcqRel);

    // `AtomicI16` test cases
    let x = AtomicI16::new(0);

    let _ = x.load(Ordering::Acquire);
    let _ = x.load(Ordering::SeqCst);
    let _ = x.load(Ordering::Relaxed);
    let _ = x.load(Ordering::Release);
    let _ = x.load(Ordering::AcqRel);

    x.store(1, Ordering::Release);
    x.store(1, Ordering::SeqCst);
    x.store(1, Ordering::Relaxed);
    x.store(1, Ordering::Acquire);
    x.store(1, Ordering::AcqRel);

    // `AtomicI32` test cases
    let x = AtomicI32::new(0);

    let _ = x.load(Ordering::Acquire);
    let _ = x.load(Ordering::SeqCst);
    let _ = x.load(Ordering::Relaxed);
    let _ = x.load(Ordering::Release);
    let _ = x.load(Ordering::AcqRel);

    x.store(1, Ordering::Release);
    x.store(1, Ordering::SeqCst);
    x.store(1, Ordering::Relaxed);
    x.store(1, Ordering::Acquire);
    x.store(1, Ordering::AcqRel);

    // `AtomicI64` test cases
    let x = AtomicI64::new(0);

    let _ = x.load(Ordering::Acquire);
    let _ = x.load(Ordering::SeqCst);
    let _ = x.load(Ordering::Relaxed);
    let _ = x.load(Ordering::Release);
    let _ = x.load(Ordering::AcqRel);

    x.store(1, Ordering::Release);
    x.store(1, Ordering::SeqCst);
    x.store(1, Ordering::Relaxed);
    x.store(1, Ordering::Acquire);
    x.store(1, Ordering::AcqRel);

    // `AtomicIsize` test cases
    let x = AtomicIsize::new(0);

    let _ = x.load(Ordering::Acquire);
    let _ = x.load(Ordering::SeqCst);
    let _ = x.load(Ordering::Relaxed);
    let _ = x.load(Ordering::Release);
    let _ = x.load(Ordering::AcqRel);

    x.store(1, Ordering::Release);
    x.store(1, Ordering::SeqCst);
    x.store(1, Ordering::Relaxed);
    x.store(1, Ordering::Acquire);
    x.store(1, Ordering::AcqRel);

    // `AtomicPtr` test cases
    let ptr = &mut 5;
    let other_ptr = &mut 10;
    let x = AtomicPtr::new(ptr);

    let _ = x.load(Ordering::Acquire);
    let _ = x.load(Ordering::SeqCst);
    let _ = x.load(Ordering::Relaxed);
    let _ = x.load(Ordering::Release);
    let _ = x.load(Ordering::AcqRel);

    x.store(other_ptr, Ordering::Release);
    x.store(other_ptr, Ordering::SeqCst);
    x.store(other_ptr, Ordering::Relaxed);
    x.store(other_ptr, Ordering::Acquire);
    x.store(other_ptr, Ordering::AcqRel);

    // `AtomicU8` test cases
    let x = AtomicU8::new(0);

    let _ = x.load(Ordering::Acquire);
    let _ = x.load(Ordering::SeqCst);
    let _ = x.load(Ordering::Relaxed);
    let _ = x.load(Ordering::Release);
    let _ = x.load(Ordering::AcqRel);

    x.store(1, Ordering::Release);
    x.store(1, Ordering::SeqCst);
    x.store(1, Ordering::Relaxed);
    x.store(1, Ordering::Acquire);
    x.store(1, Ordering::AcqRel);

    // `AtomicU16` test cases
    let x = AtomicU16::new(0);

    let _ = x.load(Ordering::Acquire);
    let _ = x.load(Ordering::SeqCst);
    let _ = x.load(Ordering::Relaxed);
    let _ = x.load(Ordering::Release);
    let _ = x.load(Ordering::AcqRel);

    x.store(1, Ordering::Release);
    x.store(1, Ordering::SeqCst);
    x.store(1, Ordering::Relaxed);
    x.store(1, Ordering::Acquire);
    x.store(1, Ordering::AcqRel);

    // `AtomicU32` test cases
    let x = AtomicU32::new(0);

    let _ = x.load(Ordering::Acquire);
    let _ = x.load(Ordering::SeqCst);
    let _ = x.load(Ordering::Relaxed);
    let _ = x.load(Ordering::Release);
    let _ = x.load(Ordering::AcqRel);

    x.store(1, Ordering::Release);
    x.store(1, Ordering::SeqCst);
    x.store(1, Ordering::Relaxed);
    x.store(1, Ordering::Acquire);
    x.store(1, Ordering::AcqRel);

    // `AtomicU64` test cases
    let x = AtomicU64::new(0);

    let _ = x.load(Ordering::Acquire);
    let _ = x.load(Ordering::SeqCst);
    let _ = x.load(Ordering::Relaxed);
    let _ = x.load(Ordering::Release);
    let _ = x.load(Ordering::AcqRel);

    x.store(1, Ordering::Release);
    x.store(1, Ordering::SeqCst);
    x.store(1, Ordering::Relaxed);
    x.store(1, Ordering::Acquire);
    x.store(1, Ordering::AcqRel);

    // `AtomicUsize` test cases
    let x = AtomicUsize::new(0);

    let _ = x.load(Ordering::Acquire);
    let _ = x.load(Ordering::SeqCst);
    let _ = x.load(Ordering::Relaxed);
    let _ = x.load(Ordering::Release);
    let _ = x.load(Ordering::AcqRel);

    x.store(1, Ordering::Release);
    x.store(1, Ordering::SeqCst);
    x.store(1, Ordering::Relaxed);
    x.store(1, Ordering::Acquire);
    x.store(1, Ordering::AcqRel);
}
