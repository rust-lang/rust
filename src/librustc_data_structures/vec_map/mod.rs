#[cfg(test)]
mod test;

/// A (multi-)map based on a sorted vector. This uses binary search to
/// find the starting index for a given element and can be a fairly
/// efficient data structure, particularly for small-ish sets of data.
///
/// To use, you supply the starting vector along with a "key fn" that
/// extracts the key from each element.
pub struct VecMap<E, KeyFn> {
    data: Vec<E>,
    key_fn: KeyFn,
}

impl<E, K, KeyFn> VecMap<E, KeyFn>
where
    KeyFn: Fn(&E) -> K,
    K: Ord + std::fmt::Debug,
{
    pub fn new(
        mut data: Vec<E>,
        key_fn: KeyFn,
    ) -> Self {
        data.sort_by_key(&key_fn);
        Self { data, key_fn }
    }

    /// Extract the first index for the given key using binary search.
    /// Returns `None` if there is no such index.
    fn get_range(&self, key: &K) -> Option<(usize, usize)> {
        match self.data.binary_search_by_key(key, &self.key_fn) {
            Ok(mid) => {
                // We get back *some* element with the given key -- so
                // search backwards to find the *first* one.
                //
                // (It'd be more efficient to use a "galloping" search
                // here, but it's not really worth it for small-ish
                // amounts of data.)
                let mut start = mid;
                while start > 0 {
                    if (self.key_fn)(&self.data[start - 1]) == *key {
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
                let max = self.data.len();
                while end < max {
                    if (self.key_fn)(&self.data[end]) == *key {
                        end += 1;
                    } else {
                        break;
                    }
                }

                Some((start, end))
            }
            Err(_) => None,
        }
    }

    /// Gets the (first) value associated with a given key.
    pub fn get_first(&self, key: &K) -> Option<&E> {
        let (start, _) = self.get_range(key)?;
        Some(&self.data[start])
    }

    /// Gets a slice of values associated with the given key.
    pub fn get_all(&self, key: &K) -> &[E] {
        let (start, end) = self.get_range(key).unwrap_or((0, 0));
        &self.data[start..end]
    }

    /// Gets a slice of values associated with the given key.
    pub fn get_iter<'k>(&'k self, key: &'k K) -> impl Iterator<Item = &'k E> {
        self.get_all(key).iter()
    }
}
