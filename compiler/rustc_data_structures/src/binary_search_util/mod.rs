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
    let mid = match data.binary_search_by_key(key, &key_fn) {
        Ok(mid) => mid,
        Err(_) => return &[],
    };
    let size = data.len();

    // We get back *some* element with the given key -- so do
    // a galloping search backwards to find the *first* one.
    let mut start = mid;
    let mut previous = mid;
    let mut step = 1;
    loop {
        start = start.saturating_sub(step);
        if start == 0 || key_fn(&data[start]) != *key {
            break;
        }
        previous = start;
        step *= 2;
    }
    step = previous - start;
    while step > 1 {
        let half = step / 2;
        let mid = start + half;
        if key_fn(&data[mid]) != *key {
            start = mid;
        }
        step -= half;
    }
    // adjust by one if we have overshot
    if start < size && key_fn(&data[start]) != *key {
        start += 1;
    }

    // Now search forward to find the *last* one.
    let mut end = mid;
    let mut previous = mid;
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
