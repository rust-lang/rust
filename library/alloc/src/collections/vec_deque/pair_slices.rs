use core::cmp::{self};
use core::mem::replace;

use crate::alloc::Allocator;

use super::VecDeque;

/// PairSlices pairs up equal length slice parts of two deques
///
/// For example, given deques "A" and "B" with the following division into slices:
///
/// A: [0 1 2] [3 4 5]
/// B: [a b] [c d e]
///
/// It produces the following sequence of matching slices:
///
/// ([0 1], [a b])
/// (\[2\], \[c\])
/// ([3 4], [d e])
///
/// and the uneven remainder of either A or B is skipped.
pub struct PairSlices<'a, 'b, T> {
    a0: &'a mut [T],
    a1: &'a mut [T],
    b0: &'b [T],
    b1: &'b [T],
}

impl<'a, 'b, T> PairSlices<'a, 'b, T> {
    pub fn from<A: Allocator>(to: &'a mut VecDeque<T, A>, from: &'b VecDeque<T, A>) -> Self {
        let (a0, a1) = to.as_mut_slices();
        let (b0, b1) = from.as_slices();
        PairSlices { a0, a1, b0, b1 }
    }

    pub fn has_remainder(&self) -> bool {
        !self.b0.is_empty()
    }

    pub fn remainder(self) -> impl Iterator<Item = &'b [T]> {
        IntoIterator::into_iter([self.b0, self.b1])
    }
}

impl<'a, 'b, T> Iterator for PairSlices<'a, 'b, T> {
    type Item = (&'a mut [T], &'b [T]);
    fn next(&mut self) -> Option<Self::Item> {
        // Get next part length
        let part = cmp::min(self.a0.len(), self.b0.len());
        if part == 0 {
            return None;
        }
        let (p0, p1) = replace(&mut self.a0, &mut []).split_at_mut(part);
        let (q0, q1) = self.b0.split_at(part);

        // Move a1 into a0, if it's empty (and b1, b0 the same way).
        self.a0 = p1;
        self.b0 = q1;
        if self.a0.is_empty() {
            self.a0 = replace(&mut self.a1, &mut []);
        }
        if self.b0.is_empty() {
            self.b0 = replace(&mut self.b1, &[]);
        }
        Some((p0, q0))
    }
}
