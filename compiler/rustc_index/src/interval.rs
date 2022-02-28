use std::iter::Step;
use std::marker::PhantomData;
use std::ops::Bound;
use std::ops::RangeBounds;

use crate::vec::Idx;
use crate::vec::IndexVec;
use rustc_macros::{Decodable, Encodable};
use smallvec::SmallVec;

#[cfg(test)]
mod tests;

/// Stores a set of intervals on the indices.
#[derive(Clone, PartialEq, Eq, Hash, Encodable, Decodable)]
pub struct IntervalSet<I> {
    // Start, end
    map: SmallVec<[(I, I); 4]>,
    domain: usize,
    _data: PhantomData<I>,
}

impl<I: Ord + Idx + Step> std::fmt::Debug for IntervalSet<I> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        struct AsList<'a, I>(&'a IntervalSet<I>);

        impl<'a, I: Idx + Ord + Step> std::fmt::Debug for AsList<'a, I> {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.debug_list().entries(self.0.iter_intervals()).finish()
            }
        }

        let mut s = f.debug_struct("IntervalSet");
        s.field("domain_size", &self.domain);
        s.field("set", &AsList(&self));
        Ok(())
    }
}

#[inline]
fn inclusive_start<T: Idx>(range: impl RangeBounds<T>) -> T {
    match range.start_bound() {
        Bound::Included(start) => *start,
        Bound::Excluded(start) => T::new(start.index() + 1),
        Bound::Unbounded => T::new(0),
    }
}

#[inline]
fn inclusive_end<T: Idx>(domain: usize, range: impl RangeBounds<T>) -> Option<T> {
    let end = match range.end_bound() {
        Bound::Included(end) => *end,
        Bound::Excluded(end) => T::new(end.index().checked_sub(1)?),
        Bound::Unbounded => T::new(domain.checked_sub(1)?),
    };
    Some(end)
}

impl<I: Ord + Idx> IntervalSet<I> {
    pub fn new(domain: usize) -> IntervalSet<I> {
        IntervalSet { map: SmallVec::new(), domain, _data: PhantomData }
    }

    /// Ensure that the set's domain is at least `min_domain_size`.
    pub fn ensure(&mut self, min_domain_size: usize) {
        if self.domain < min_domain_size {
            self.domain = min_domain_size;
        }
    }

    pub fn domain_size(&self) -> usize {
        self.domain
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
        self.map.iter().map(|&(start, end)| start..I::new(end.index() + 1))
    }

    /// Returns true if we increased the number of elements present.
    pub fn insert(&mut self, point: I) -> bool {
        self.insert_range(point..=point)
    }

    pub fn remove(&mut self, point: I) {
        self.remove_range(point..=point);
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
            let next = self.map.partition_point(|r| r.0.index() <= end.index() + 1);
            if let Some(last) = next.checked_sub(1) {
                let (prev_start, prev_end) = &mut self.map[last];
                if prev_end.index() + 1 >= start.index() {
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

    pub fn remove_range(&mut self, range: impl RangeBounds<I> + Clone) {
        let start = inclusive_start(range.clone());
        let Some(end) = inclusive_end(self.domain, range.clone()) else {
            // empty range
            return;
        };
        if start > end {
            return;
        }
        // We insert the range, so that any previous gaps are merged into just one large
        // range, which we can then split in the next step (either inserting a
        // smaller range after or not).
        self.insert_range(range);
        // Find the range we just inserted.
        let idx = self.map.partition_point(|r| r.0 <= end).checked_sub(1).unwrap();
        let (prev_start, prev_end) = self.map.remove(idx);
        // The range we're looking at contains the range we're removing completely.
        assert!(prev_start <= start && end <= prev_end);
        self.insert_range(prev_start..start);
        self.insert_range((Bound::Excluded(end), Bound::Included(prev_end)));
    }

    pub fn contains(&self, needle: I) -> bool {
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
        // FIXME: Performance here is probably not great. We will be doing a lot
        // of pointless tree traversals.
        other.iter().all(|elem| self.contains(elem))
    }

    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    /// Returns the minimum (first) element present in the set from `range`.
    pub fn first_set_in(&self, range: impl RangeBounds<I> + Clone) -> Option<I> {
        let start = inclusive_start(range.clone());
        let Some(end) = inclusive_end(self.domain, range) else {
            // empty range
            return None;
        };
        if start > end {
            return None;
        }
        let range = self.map.get(self.map.partition_point(|r| r.1 < start))?;
        if range.0 > end { None } else { Some(std::cmp::max(range.0, start)) }
    }

    /// Returns the minimum (first) element **not** present in the set from `range`.
    pub fn first_gap_in(&self, range: impl RangeBounds<I> + Clone) -> Option<I> {
        let start = inclusive_start(range.clone());
        let Some(end) = inclusive_end(self.domain, range) else {
            // empty range
            return None;
        };
        if start > end {
            return None;
        }
        let Some(range) = self.map.get(self.map.partition_point(|r| r.1 < start)) else {
            return Some(start);
        };
        if start < range.0 {
            return Some(start);
        } else if range.1.index() + 1 < self.domain {
            if range.1.index() + 1 <= end.index() {
                return Some(I::new(range.1.index() + 1));
            }
        }

        None
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
        if start <= *prev_end { Some(std::cmp::min(*prev_end, end)) } else { None }
    }

    pub fn insert_all(&mut self) {
        self.clear();
        self.map.push((I::new(0), I::new(self.domain)));
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

impl<R: Idx, C: Ord + Step + Idx> SparseIntervalMatrix<R, C> {
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
