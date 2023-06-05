//! Pattern utilities.
//!
//! Originates from `rustc_hir::pat_util`

use std::iter::{Enumerate, ExactSizeIterator};

pub(crate) struct EnumerateAndAdjust<I> {
    enumerate: Enumerate<I>,
    gap_pos: usize,
    gap_len: usize,
}

impl<I> Iterator for EnumerateAndAdjust<I>
where
    I: Iterator,
{
    type Item = (usize, <I as Iterator>::Item);

    fn next(&mut self) -> Option<(usize, <I as Iterator>::Item)> {
        self.enumerate
            .next()
            .map(|(i, elem)| (if i < self.gap_pos { i } else { i + self.gap_len }, elem))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.enumerate.size_hint()
    }
}

pub(crate) trait EnumerateAndAdjustIterator {
    fn enumerate_and_adjust(
        self,
        expected_len: usize,
        gap_pos: Option<usize>,
    ) -> EnumerateAndAdjust<Self>
    where
        Self: Sized;
}

impl<T: ExactSizeIterator> EnumerateAndAdjustIterator for T {
    fn enumerate_and_adjust(
        self,
        expected_len: usize,
        gap_pos: Option<usize>,
    ) -> EnumerateAndAdjust<Self>
    where
        Self: Sized,
    {
        let actual_len = self.len();
        EnumerateAndAdjust {
            enumerate: self.enumerate(),
            gap_pos: gap_pos.unwrap_or(expected_len),
            gap_len: expected_len - actual_len,
        }
    }
}
