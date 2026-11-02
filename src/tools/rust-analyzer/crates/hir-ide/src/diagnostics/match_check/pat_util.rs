//! Pattern utilities.
//!
//! Originates from `rustc_hir::pat_util`

use std::iter::Enumerate;

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
    /// When there is a list of items with a gap of an unknown length inside, and another list
    /// of item it should be zipped against, this operates on the list with the gap and returns,
    /// for each item, the index it should match in the other list.
    ///
    /// When compiling Rust, such situation often occurs for tuple structs/tuples with a rest pattern
    /// that should be matched against the fields.
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
