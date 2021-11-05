use std::iter::Step;
use std::marker::PhantomData;
use std::ops::Bound;
use std::ops::RangeBounds;

use crate::vec::Idx;
use crate::vec::IndexVec;
use smallvec::SmallVec;

#[cfg(test)]
mod tests;

/// Stores a set of intervals on the indices.
#[derive(Debug, Clone)]
pub struct IntervalSet<I> {
    // Start, end
    map: SmallVec<[(u32, u32); 4]>,
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

    pub fn iter(&self) -> impl Iterator<Item = I> + '_
    where
        I: Step,
    {
        self.iter_intervals().flatten()
    }

    /// Iterates through intervals stored in the set, in order.
    pub fn iter_intervals(&self) -> impl Iterator<Item = std::ops::Range<I>> + '_
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
        let Some(mut end) = inclusive_end(self.domain, range) else {
            // empty range
            return false;
        };
        if start > end {
            return false;
        }

        loop {
            // This condition looks a bit weird, but actually makes sense.
            //
            // if r.0 == end + 1, then we're actually adjacent, so we want to
            // continue to the next range. We're looking here for the first
            // range which starts *non-adjacently* to our end.
            let next = self.map.partition_point(|r| r.0 <= end + 1);
            if let Some(last) = next.checked_sub(1) {
                let (prev_start, prev_end) = &mut self.map[last];
                if *prev_end + 1 >= start {
                    // If the start for the inserted range is adjacent to the
                    // end of the previous, we can extend the previous range.
                    if start < *prev_start {
                        // Our range starts before the one we found. We'll need
                        // to *remove* it, and then try again.
                        //
                        // FIXME: This is not so efficient; we may need to
                        // recurse a bunch of times here. Instead, it's probably
                        // better to do something like drain_filter(...) on the
                        // map to be able to delete or modify all the ranges in
                        // start..=end and then potentially re-insert a new
                        // range.
                        end = std::cmp::max(end, *prev_end);
                        self.map.remove(last);
                    } else {
                        // We overlap with the previous range, increase it to
                        // include us.
                        //
                        // Make sure we're actually going to *increase* it though --
                        // it may be that end is just inside the previously existing
                        // set.
                        return if end > *prev_end {
                            *prev_end = end;
                            true
                        } else {
                            false
                        };
                    }
                } else {
                    // Otherwise, we don't overlap, so just insert
                    self.map.insert(last + 1, (start, end));
                    return true;
                }
            } else {
                if self.map.is_empty() {
                    // Quite common in practice, and expensive to call memcpy
                    // with length zero.
                    self.map.push((start, end));
                } else {
                    self.map.insert(next, (start, end));
                }
                return true;
            }
        }
    }

    pub fn contains(&self, needle: I) -> bool {
        let needle = needle.index() as u32;
        let last = match self.map.partition_point(|r| r.0 <= needle).checked_sub(1) {
            Some(idx) => idx,
            None => {
                // All ranges in the map start after the new range's end
                return false;
            }
        };
        let (_, prev_end) = &self.map[last];
        needle <= *prev_end
    }

    pub fn superset(&self, other: &IntervalSet<I>) -> bool
    where
        I: Step,
    {
        // FIXME: Performance here is probably not great. We will be doing a lot
        // of pointless tree traversals.
        other.iter().all(|elem| self.contains(elem))
    }

    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
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
        let last = match self.map.partition_point(|r| r.0 <= end).checked_sub(1) {
            Some(idx) => idx,
            None => {
                // All ranges in the map start after the new range's end
                return None;
            }
        };
        let (_, prev_end) = &self.map[last];
        if start <= *prev_end { Some(I::new(std::cmp::min(*prev_end, end) as usize)) } else { None }
    }

    pub fn insert_all(&mut self) {
        self.clear();
        self.map.push((0, self.domain.try_into().unwrap()));
    }

    pub fn union(&mut self, other: &IntervalSet<I>) -> bool
    where
        I: Step,
    {
        assert_eq!(self.domain, other.domain);
        let mut did_insert = false;
        for range in other.iter_intervals() {
            did_insert |= self.insert_range(range);
        }
        did_insert
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
        self.rows.ensure_contains_elem(row, || IntervalSet::new(self.column_size));
        &mut self.rows[row]
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

    pub fn contains(&self, row: R, point: C) -> bool {
        self.row(row).map_or(false, |r| r.contains(point))
    }
}
