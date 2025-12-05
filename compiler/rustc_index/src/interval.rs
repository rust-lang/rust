use std::iter::Step;
use std::marker::PhantomData;
use std::ops::{Bound, Range, RangeBounds};

use smallvec::SmallVec;

use crate::idx::Idx;
use crate::vec::IndexVec;

#[cfg(test)]
mod tests;

/// Stores a set of intervals on the indices.
///
/// The elements in `map` are sorted and non-adjacent, which means
/// the second value of the previous element is *greater* than the
/// first value of the following element.
#[derive(Debug, Clone)]
pub struct IntervalSet<I> {
    // Start, end (both inclusive)
    map: SmallVec<[(u32, u32); 2]>,
    domain: usize,
    _data: PhantomData<I>,
}

#[inline]
fn inclusive_start<T: Idx>(range: impl RangeBounds<T>) -> u32 {
    match range.start_bound() {
        Bound::Included(start) => start.index() as u32,
        Bound::Excluded(start) => start.index() as u32 + 1,
        Bound::Unbounded => 0,
    }
}

#[inline]
fn inclusive_end<T: Idx>(domain: usize, range: impl RangeBounds<T>) -> Option<u32> {
    let end = match range.end_bound() {
        Bound::Included(end) => end.index() as u32,
        Bound::Excluded(end) => end.index().checked_sub(1)? as u32,
        Bound::Unbounded => domain.checked_sub(1)? as u32,
    };
    Some(end)
}

impl<I: Idx> IntervalSet<I> {
    pub fn new(domain: usize) -> IntervalSet<I> {
        IntervalSet { map: SmallVec::new(), domain, _data: PhantomData }
    }

    pub fn clear(&mut self) {
        self.map.clear();
    }

    pub fn iter(&self) -> impl Iterator<Item = I>
    where
        I: Step,
    {
        self.iter_intervals().flatten()
    }

    /// Iterates through intervals stored in the set, in order.
    pub fn iter_intervals(&self) -> impl Iterator<Item = std::ops::Range<I>>
    where
        I: Step,
    {
        self.map.iter().map(|&(start, end)| I::new(start as usize)..I::new(end as usize + 1))
    }

    /// Returns true if we increased the number of elements present.
    pub fn insert(&mut self, point: I) -> bool {
        self.insert_range(point..=point)
    }

    /// Returns true if we increased the number of elements present.
    pub fn insert_range(&mut self, range: impl RangeBounds<I> + Clone) -> bool {
        let start = inclusive_start(range.clone());
        let Some(end) = inclusive_end(self.domain, range) else {
            // empty range
            return false;
        };
        if start > end {
            return false;
        }

        // This condition looks a bit weird, but actually makes sense.
        //
        // if r.0 == end + 1, then we're actually adjacent, so we want to
        // continue to the next range. We're looking here for the first
        // range which starts *non-adjacently* to our end.
        let next = self.map.partition_point(|r| r.0 <= end + 1);
        let result = if let Some(right) = next.checked_sub(1) {
            let (prev_start, prev_end) = self.map[right];
            if prev_end + 1 >= start {
                // If the start for the inserted range is adjacent to the
                // end of the previous, we can extend the previous range.
                if start < prev_start {
                    // The first range which ends *non-adjacently* to our start.
                    // And we can ensure that left <= right.
                    let left = self.map.partition_point(|l| l.1 + 1 < start);
                    let min = std::cmp::min(self.map[left].0, start);
                    let max = std::cmp::max(prev_end, end);
                    self.map[right] = (min, max);
                    if left != right {
                        self.map.drain(left..right);
                    }
                    true
                } else {
                    // We overlap with the previous range, increase it to
                    // include us.
                    //
                    // Make sure we're actually going to *increase* it though --
                    // it may be that end is just inside the previously existing
                    // set.
                    if end > prev_end {
                        self.map[right].1 = end;
                        true
                    } else {
                        false
                    }
                }
            } else {
                // Otherwise, we don't overlap, so just insert
                self.map.insert(right + 1, (start, end));
                true
            }
        } else {
            if self.map.is_empty() {
                // Quite common in practice, and expensive to call memcpy
                // with length zero.
                self.map.push((start, end));
            } else {
                self.map.insert(next, (start, end));
            }
            true
        };
        debug_assert!(
            self.check_invariants(),
            "wrong intervals after insert {start:?}..={end:?} to {self:?}"
        );
        result
    }

    /// Specialized version of `insert` when we know that the inserted point is *after* any
    /// contained.
    pub fn append(&mut self, point: I) {
        let point = point.index() as u32;

        if let Some((_, last_end)) = self.map.last_mut() {
            assert!(*last_end <= point);
            if point == *last_end {
                // The point is already in the set.
            } else if point == *last_end + 1 {
                *last_end = point;
            } else {
                self.map.push((point, point));
            }
        } else {
            self.map.push((point, point));
        }

        debug_assert!(
            self.check_invariants(),
            "wrong intervals after append {point:?} to {self:?}"
        );
    }

    pub fn contains(&self, needle: I) -> bool {
        let needle = needle.index() as u32;
        let Some(last) = self.map.partition_point(|r| r.0 <= needle).checked_sub(1) else {
            // All ranges in the map start after the new range's end
            return false;
        };
        let (_, prev_end) = &self.map[last];
        needle <= *prev_end
    }

    pub fn superset(&self, other: &IntervalSet<I>) -> bool
    where
        I: Step,
    {
        let mut sup_iter = self.iter_intervals();
        let mut current = None;
        let contains = |sup: Range<I>, sub: Range<I>, current: &mut Option<Range<I>>| {
            if sup.end < sub.start {
                // if `sup.end == sub.start`, the next sup doesn't contain `sub.start`
                None // continue to the next sup
            } else if sup.end >= sub.end && sup.start <= sub.start {
                *current = Some(sup); // save the current sup
                Some(true)
            } else {
                Some(false)
            }
        };
        other.iter_intervals().all(|sub| {
            current
                .take()
                .and_then(|sup| contains(sup, sub.clone(), &mut current))
                .or_else(|| sup_iter.find_map(|sup| contains(sup, sub.clone(), &mut current)))
                .unwrap_or(false)
        })
    }

    pub fn disjoint(&self, other: &IntervalSet<I>) -> bool
    where
        I: Step,
    {
        let helper = move || {
            let mut self_iter = self.iter_intervals();
            let mut other_iter = other.iter_intervals();

            let mut self_current = self_iter.next()?;
            let mut other_current = other_iter.next()?;

            loop {
                if self_current.end <= other_current.start {
                    self_current = self_iter.next()?;
                    continue;
                }
                if other_current.end <= self_current.start {
                    other_current = other_iter.next()?;
                    continue;
                }
                return Some(false);
            }
        };
        helper().unwrap_or(true)
    }

    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    /// Equivalent to `range.iter().find(|i| !self.contains(i))`.
    pub fn first_unset_in(&self, range: impl RangeBounds<I> + Clone) -> Option<I> {
        let start = inclusive_start(range.clone());
        let Some(end) = inclusive_end(self.domain, range) else {
            // empty range
            return None;
        };
        if start > end {
            return None;
        }
        let Some(last) = self.map.partition_point(|r| r.0 <= start).checked_sub(1) else {
            // All ranges in the map start after the new range's end
            return Some(I::new(start as usize));
        };
        let (_, prev_end) = self.map[last];
        if start > prev_end {
            Some(I::new(start as usize))
        } else if prev_end < end {
            Some(I::new(prev_end as usize + 1))
        } else {
            None
        }
    }

    /// Returns the maximum (last) element present in the set from `range`.
    pub fn last_set_in(&self, range: impl RangeBounds<I> + Clone) -> Option<I> {
        let start = inclusive_start(range.clone());
        let Some(end) = inclusive_end(self.domain, range) else {
            // empty range
            return None;
        };
        if start > end {
            return None;
        }
        let Some(last) = self.map.partition_point(|r| r.0 <= end).checked_sub(1) else {
            // All ranges in the map start after the new range's end
            return None;
        };
        let (_, prev_end) = &self.map[last];
        if start <= *prev_end { Some(I::new(std::cmp::min(*prev_end, end) as usize)) } else { None }
    }

    pub fn insert_all(&mut self) {
        self.clear();
        if let Some(end) = self.domain.checked_sub(1) {
            self.map.push((0, end.try_into().unwrap()));
        }
        debug_assert!(self.check_invariants());
    }

    pub fn union(&mut self, other: &IntervalSet<I>) -> bool
    where
        I: Step,
    {
        assert_eq!(self.domain, other.domain);
        if self.map.len() < other.map.len() {
            let backup = self.clone();
            self.map.clone_from(&other.map);
            return self.union(&backup);
        }

        let mut did_insert = false;
        for range in other.iter_intervals() {
            did_insert |= self.insert_range(range);
        }
        debug_assert!(self.check_invariants());
        did_insert
    }

    // Check the intervals are valid, sorted and non-adjacent
    fn check_invariants(&self) -> bool {
        let mut current: Option<u32> = None;
        for (start, end) in &self.map {
            if start > end || current.is_some_and(|x| x + 1 >= *start) {
                return false;
            }
            current = Some(*end);
        }
        current.is_none_or(|x| x < self.domain as u32)
    }
}

/// This data structure optimizes for cases where the stored bits in each row
/// are expected to be highly contiguous (long ranges of 1s or 0s), in contrast
/// to BitMatrix and SparseBitMatrix which are optimized for
/// "random"/non-contiguous bits and cheap(er) point queries at the expense of
/// memory usage.
#[derive(Clone)]
pub struct SparseIntervalMatrix<R, C>
where
    R: Idx,
    C: Idx,
{
    rows: IndexVec<R, IntervalSet<C>>,
    column_size: usize,
}

impl<R: Idx, C: Step + Idx> SparseIntervalMatrix<R, C> {
    pub fn new(column_size: usize) -> SparseIntervalMatrix<R, C> {
        SparseIntervalMatrix { rows: IndexVec::new(), column_size }
    }

    pub fn rows(&self) -> impl Iterator<Item = R> {
        self.rows.indices()
    }

    pub fn row(&self, row: R) -> Option<&IntervalSet<C>> {
        self.rows.get(row)
    }

    fn ensure_row(&mut self, row: R) -> &mut IntervalSet<C> {
        self.rows.ensure_contains_elem(row, || IntervalSet::new(self.column_size))
    }

    pub fn union_row(&mut self, row: R, from: &IntervalSet<C>) -> bool
    where
        C: Step,
    {
        self.ensure_row(row).union(from)
    }

    pub fn union_rows(&mut self, read: R, write: R) -> bool
    where
        C: Step,
    {
        if read == write || self.rows.get(read).is_none() {
            return false;
        }
        self.ensure_row(write);
        let (read_row, write_row) = self.rows.pick2_mut(read, write);
        write_row.union(read_row)
    }

    pub fn insert_all_into_row(&mut self, row: R) {
        self.ensure_row(row).insert_all();
    }

    pub fn insert_range(&mut self, row: R, range: impl RangeBounds<C> + Clone) {
        self.ensure_row(row).insert_range(range);
    }

    pub fn insert(&mut self, row: R, point: C) -> bool {
        self.ensure_row(row).insert(point)
    }

    pub fn append(&mut self, row: R, point: C) {
        self.ensure_row(row).append(point)
    }

    pub fn contains(&self, row: R, point: C) -> bool {
        self.row(row).is_some_and(|r| r.contains(point))
    }
}
