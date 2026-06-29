/// Represents a set of `Size` values as a sorted list of ranges.
///
/// These are (offset, length) pairs, and they are sorted and mutually disjoint,
/// and never adjacent (i.e. there's always a gap between two of them).
#[derive(Debug, Clone)]
pub struct RangeSet<T>(pub Vec<(T, T)>);

impl<T> RangeSet<T>
where
    T: Copy + Ord + Default,
    T: core::ops::Add<Output = T>,
    T: core::ops::Sub<Output = T>,
{
    pub fn new() -> Self {
        Self(Vec::new())
    }

    pub fn add_range(&mut self, offset: T, size: T) {
        if size == T::default() {
            // No need to track empty ranges.
            return;
        }
        let v = &mut self.0;
        // We scan for a partition point where the left partition is all the elements that end
        // strictly before we start. Those are elements that are too "low" to merge with us.
        let idx =
            v.partition_point(|&(other_offset, other_size)| other_offset + other_size < offset);
        // Now we want to either merge with the first element of the second partition, or insert ourselves before that.
        if let Some(&(other_offset, other_size)) = v.get(idx)
            && offset + size >= other_offset
        {
            // Their end is >= our start (otherwise it would not be in the 2nd partition) and
            // our end is >= their start. This means we can merge the ranges.
            let new_start = other_offset.min(offset);
            let mut new_end = (other_offset + other_size).max(offset + size);
            // We grew to the right, so merge with overlapping/adjacent elements.
            // (We also may have grown to the left, but that can never make us adjacent with
            // anything there since we selected the first such candidate via `partition_point`.)
            let mut scan_right = 1;
            while let Some(&(next_offset, next_size)) = v.get(idx + scan_right)
                && new_end >= next_offset
            {
                // Increase our size to absorb the next element.
                new_end = new_end.max(next_offset + next_size);
                // Look at the next element.
                scan_right += 1;
            }
            // Update the element we grew.
            v[idx] = (new_start, new_end - new_start);
            // Remove the elements we absorbed (if any).
            if scan_right > 1 {
                drop(v.drain((idx + 1)..(idx + scan_right)));
            }
        } else {
            // Insert new element.
            v.insert(idx, (offset, size));
        }
    }
}
