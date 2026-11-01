use std::sync::atomic::Ordering::Relaxed;
use std::sync::atomic::*;

unsafe fn split_atomic(a: &AtomicU16) -> &[AtomicU8; 2] {
    unsafe { std::mem::transmute(a) }
}

fn main() {
    std::thread::spawn(move || {
        let a: AtomicU16 = 0.into();

        // Bump our vector clock so the non-atomic initializing write above is separate
        // from the first relaxed store below. Or something like that.
        AtomicU16::new(0).store(0, Relaxed);

        a.store(0, Relaxed);
        let parts = unsafe { split_atomic(&a) };
        parts[0].store(0, Relaxed);
    })
    .join()
    .unwrap();
}
