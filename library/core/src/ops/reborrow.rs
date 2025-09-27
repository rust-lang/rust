/// Allows value to be reborrowed as exclusive, creating a copy of the value
/// that disables the source for reads and writes for the lifetime of the copy.
#[lang = "reborrow"]
#[unstable(feature = "reborrow", issue = "145612")]
pub trait Reborrow {
    // Empty.
}

/// Allows reborrowable value to be reborrowed as shared, creating a copy
/// that disables the source for writes for the lifetime of the copy.
#[lang = "coerce_shared"]
#[unstable(feature = "reborrow", issue = "145612")]
pub trait CoerceShared: Reborrow {
    /// The type of this value when reborrowed as shared.
    type Target: Copy;
}
