#[cfg(test)]
mod test;

/// Uses a sorted slice `data: &[E]` as a kind of "multi-map". The
/// `key_fn` extracts a key of type `K` from the data, and this
/// function finds the range of elements that match the key. `data`
/// must have been sorted as if by a call to `sort_by_key` for this to
/// work.
pub fn binary_search_slice<E, K>(data: &'d [E], key_fn: impl Fn(&E) -> K, key: &K) -> &'d [E]
where
    K: Ord,
{
    let mid = match data.binary_search_by_key(key, &key_fn) {
        Ok(mid) => mid,
        Err(_) => return &[],
    };

    // We get back *some* element with the given key -- so
    // search backwards to find the *first* one.
    //
    // (It'd be more efficient to use a "galloping" search
    // here, but it's not really worth it for small-ish
    // amounts of data.)
    let mut start = mid;
    while start > 0 {
        if key_fn(&data[start - 1]) == *key {
            start -= 1;
        } else {
            break;
        }
    }

    // Now search forward to find the *last* one.
    //
    // (It'd be more efficient to use a "galloping" search
    // here, but it's not really worth it for small-ish
    // amounts of data.)
    let mut end = mid + 1;
    let max = data.len();
    while end < max {
        if key_fn(&data[end]) == *key {
            end += 1;
        } else {
            break;
        }
    }

    &data[start..end]
}
