#[cfg(test)]
mod tests;

/// Uses a sorted slice `data: &[E]` as a kind of "multi-map". The
/// `key_fn` extracts a key of type `K` from the data, and this
/// function finds the range of elements that match the key. `data`
/// must have been sorted as if by a call to `sort_by_key` for this to
/// work.
pub fn binary_search_slice<'d, E, K>(data: &'d [E], key_fn: impl Fn(&E) -> K, key: &K) -> &'d [E]
where
    K: Ord,
{
    let size = data.len();
    let start = data.partition_point(|x| key_fn(x) < *key);
    // At this point `start` either points at the first entry with equal or
    // greater key or is equal to `size` in case all elements have smaller keys
    if start == size || key_fn(&data[start]) != *key {
        return &[];
    };

    // Now search forward to find the *last* one.
    let mut end = start;
    let mut previous = start;
    let mut step = 1;
    loop {
        end = end.saturating_add(step).min(size);
        if end == size || key_fn(&data[end]) != *key {
            break;
        }
        previous = end;
        step *= 2;
    }
    step = end - previous;
    while step > 1 {
        let half = step / 2;
        let mid = end - half;
        if key_fn(&data[mid]) != *key {
            end = mid;
        }
        step -= half;
    }

    &data[start..end]
}
