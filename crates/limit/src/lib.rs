//! limit defines a struct to enforce limits.

/// Represents a struct used to enforce a numerical limit.
pub struct Limit {
    upper_bound: usize,
}

impl Limit {
    /// Creates a new limit.
    #[inline]
    pub const fn new(upper_bound: usize) -> Self {
        Self { upper_bound }
    }

    /// Gets the underlying numeric limit.
    #[inline]
    pub const fn inner(&self) -> usize {
        self.upper_bound
    }

    /// Checks whether the given value is below the limit.
    /// Returns `Ok` when `other` is below `self`, and `Err` otherwise.
    #[inline]
    pub const fn check(&self, other: usize) -> Result<(), ()> {
        if other > self.upper_bound {
            Err(())
        } else {
            Ok(())
        }
    }
}
