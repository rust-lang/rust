#[test]
fn test_extend_reserve() {
    struct Collection {
        len: usize,
        capacity: usize,
    }

    impl Extend<i32> for Collection {
        fn extend<I: IntoIterator<Item = i32>>(&mut self, elements: I) {
            let iter = elements.into_iter();
            let (lower_bound, _) = iter.size_hint();
            let expected_len = self.len.saturating_add(lower_bound);
            if self.capacity < expected_len {
                // do the reserve
                self.capacity = expected_len;
            }
            // do the extend
            iter.into_iter().for_each(drop);
        }

        // no custom implementation of extend_reserve
    }

    let mut collection = Collection { len: 0, capacity: 0 };
    collection.extend_reserve(5);
    assert_eq!(collection.capacity, 5);
}
