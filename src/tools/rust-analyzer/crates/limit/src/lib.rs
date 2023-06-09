//! limit defines a struct to enforce limits.

#![warn(rust_2018_idioms, unused_lifetimes, semicolon_in_expressions_from_macros)]

#[cfg(feature = "tracking")]
use std::sync::atomic::AtomicUsize;

/// Represents a struct used to enforce a numerical limit.
#[derive(Debug)]
pub struct Limit {
    upper_bound: usize,
    #[cfg(feature = "tracking")]
    max: AtomicUsize,
}

impl Limit {
    /// Creates a new limit.
    #[inline]
    pub const fn new(upper_bound: usize) -> Self {
        Self {
            upper_bound,
            #[cfg(feature = "tracking")]
            max: AtomicUsize::new(0),
        }
    }

    /// Creates a new limit.
    #[inline]
    #[cfg(feature = "tracking")]
    pub const fn new_tracking(upper_bound: usize) -> Self {
        Self {
            upper_bound,
            #[cfg(feature = "tracking")]
            max: AtomicUsize::new(1),
        }
    }

    /// Gets the underlying numeric limit.
    #[inline]
    pub const fn inner(&self) -> usize {
        self.upper_bound
    }

    /// Checks whether the given value is below the limit.
    /// Returns `Ok` when `other` is below `self`, and `Err` otherwise.
    #[inline]
    pub fn check(&self, other: usize) -> Result<(), ()> {
        if other > self.upper_bound {
            Err(())
        } else {
            #[cfg(feature = "tracking")]
            loop {
                use std::sync::atomic::Ordering;
                let old_max = self.max.load(Ordering::Relaxed);
                if other <= old_max || old_max == 0 {
                    break;
                }
                if self
                    .max
                    .compare_exchange_weak(old_max, other, Ordering::Relaxed, Ordering::Relaxed)
                    .is_ok()
                {
                    eprintln!("new max: {other}");
                }
            }

            Ok(())
        }
    }
}
