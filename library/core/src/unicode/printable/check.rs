use crate::{iter, slice};

/// A compact representation of Unicode singletons.
///
/// This is basically a `&[u16]`, but represented as `&[(u8, &[u8])]`,
/// i.e. pairs of upper bytes and multiple corresponding lower bytes.
///
/// However, in order to reduce the pointer-sized overhead for each nested
/// slice, it is compacted again into `&[(u8, u8)]` with the length of the
/// lower bytes in the second byte, and a separate, contiguous `&[u8]` for
/// storing the lower bytes.
pub(super) struct Singletons {
    upper: &'static [(u8, u8)],
    lower: &'static [u8],
}

impl Singletons {
    /// Creates a new `Singletons` instance from compacted upper and lower bytes.
    ///
    /// # Panics
    ///
    /// Panics if the sum of all lengths (i.e. the second field of each pair) in `upper`
    /// is not equal to the length of `lower`.
    pub(super) const fn new(upper: &'static [(u8, u8)], lower: &'static [u8]) -> Self {
        let mut lower_count_total = 0;
        let mut i = 0;
        while i < upper.len() {
            lower_count_total += upper[i].1 as usize;
            i += 1;
        }
        assert!(
            lower_count_total == lower.len(),
            "Sum of lengths in `upper` does not match `lower` length."
        );

        Self { upper, lower }
    }

    #[inline]
    fn iter(&self) -> SingletonsIter {
        SingletonsIter { iter: self.upper.iter().cloned(), lower: self.lower, lower_start: 0 }
    }

    pub(super) fn check(&self, x: u16) -> bool {
        let [x_upper, x_lower] = x.to_be_bytes();
        for (upper, lowers) in self.iter() {
            if upper == x_upper {
                for &lower in lowers {
                    if lower == x_lower {
                        return false;
                    }
                }
            } else if x_upper < upper {
                break;
            }
        }

        true
    }
}

struct SingletonsIter {
    iter: iter::Cloned<slice::Iter<'static, (u8, u8)>>,
    lower: &'static [u8],
    lower_start: usize,
}

impl Iterator for SingletonsIter {
    type Item = (u8, &'static [u8]);

    fn next(&mut self) -> Option<Self::Item> {
        let (upper, lower_count) = self.iter.next()?;

        let lower_start = self.lower_start;
        let lower_end = lower_start + lower_count as usize;
        self.lower_start = lower_end;

        // SAFETY: The invariant for `Singletons` guarantees that the sum of all lengths
        // in `upper` must be equal to the lengths of `lower`, so `lower_end` is guaranteed
        // to be in range.
        let lowers = unsafe { self.lower.get_unchecked(lower_start..lower_end) };

        Some((upper, lowers))
    }
}

/// A compact representation of lengths.
pub(super) struct Normal(&'static [u8]);

impl Normal {
    pub(super) const fn new(normal: &'static [u8]) -> Self {
        // Invariant: Lengths greater than `0x7f` must be encoded as two bytes,
        // with the length contained in the remaining 15 bits, i.e. `0x7fff`.
        {
            let mut i = 0;
            while i < normal.len() {
                if normal[i] & 0b1000_0000 != 0 {
                    assert!(
                        i + 1 < normal.len(),
                        "Length greater than `0x7f` is not encoded as two bytes."
                    );
                    i += 2;
                } else {
                    i += 1;
                }
            }
        }

        Self(normal)
    }

    #[inline]
    fn iter(&self) -> NormalIter {
        NormalIter { iter: self.0.iter().cloned() }
    }

    pub(super) fn check(&self, mut x: u16) -> bool {
        let mut current = true;
        for len in self.iter() {
            x = if let Some(x) = x.checked_sub(len) { x } else { break };
            current = !current;
        }
        current
    }
}

struct NormalIter {
    iter: iter::Cloned<slice::Iter<'static, u8>>,
}

impl Iterator for NormalIter {
    type Item = u16;

    fn next(&mut self) -> Option<Self::Item> {
        let len = self.iter.next()?;

        Some(if len & 0b1000_0000 != 0 {
            let upper = len & 0b0111_1111;
            // SAFETY: The invariant of `Normal` guarantees that lengths are encoded
            // as two bytes if greater than `0x7f`, so there must be a next byte.
            let lower = unsafe { self.iter.next().unwrap_unchecked() };
            u16::from_be_bytes([upper, lower])
        } else {
            u16::from(len)
        })
    }
}
