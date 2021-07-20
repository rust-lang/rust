use crate::env;
use crate::sync::atomic::{AtomicU8, Ordering};

#[repr(u8)]
#[derive(Copy, Clone)]
pub enum InstantReliability {
    // 0 is the placeholder for initialization pending
    Default = 1,
    AssumeMonotonic = 2,
    AssumeBroken = 3,
}

static INSTANT_RELIABILITY: AtomicU8 = AtomicU8::new(0);

pub fn instant_reliability_override() -> InstantReliability {
    let current = INSTANT_RELIABILITY.load(Ordering::Relaxed);

    if current == 0 {
        let new = match env::var("RUST_CLOCK_ASSUME_MONOTONIC").as_deref() {
            Ok("1") => InstantReliability::AssumeMonotonic,
            Ok("0") => InstantReliability::AssumeBroken,
            Ok(_) => {
                eprintln!("unsupported value in RUST_CLOCK_ASSUME_MONOTONIC; using default");
                InstantReliability::Default
            }
            _ => InstantReliability::Default,
        };
        INSTANT_RELIABILITY.store(new as u8, Ordering::Relaxed);
        new
    } else {
        // SAFETY: The enum has a compatible layout and after initialization only values
        // cast from enum variants are stored in the atomic static.
        unsafe { crate::mem::transmute(current) }
    }
}
