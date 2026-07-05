use crate::io::SizeHint;

// =============================================================================
// Forwarding implementations

#[doc(hidden)]
#[unstable(feature = "core_io_internals", reason = "exposed only for libstd", issue = "none")]
impl<T> SizeHint for &mut T {
    #[inline]
    fn lower_bound(&self) -> usize {
        SizeHint::lower_bound(*self)
    }

    #[inline]
    fn upper_bound(&self) -> Option<usize> {
        SizeHint::upper_bound(*self)
    }
}

// =============================================================================
// In-memory buffer implementations

#[doc(hidden)]
#[unstable(feature = "core_io_internals", reason = "exposed only for libstd", issue = "none")]
impl SizeHint for &[u8] {
    #[inline]
    fn lower_bound(&self) -> usize {
        self.len()
    }

    #[inline]
    fn upper_bound(&self) -> Option<usize> {
        Some(self.len())
    }
}
