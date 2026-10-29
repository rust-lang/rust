use super::{SortedIndexMultiMap, SortedMap};

#[test]
fn test_sorted_index_multi_map() {
    let entries: Vec<_> = vec![(2, 0), (1, 0), (2, 1), (3, 0), (2, 2)];
    let set: SortedIndexMultiMap<usize, _, _> = entries.iter().copied().collect();

    // Insertion order is preserved.
    assert!(entries.iter().map(|(k, v)| (k, v)).eq(set.iter()));

    // Indexing
    for (i, expect) in entries.iter().enumerate() {
        assert_eq!(set[i], expect.1);
    }

    // `get_by_key` works.
    assert_eq!(set.get_by_key(3).copied().collect::<Vec<_>>(), vec![0]);
    assert!(set.get_by_key(4).next().is_none());

    // `contains_key` works
    assert!(set.contains_key(3));
    assert!(!set.contains_key(4));

    // `get_by_key` returns items in insertion order.
    let twos: Vec<_> = set.get_by_key_enumerated(2).collect();
    let idxs: Vec<usize> = twos.iter().map(|(i, _)| *i).collect();
    let values: Vec<usize> = twos.iter().map(|&(_, &v)| v).collect();

    assert_eq!(idxs, vec![0, 2, 4]);
    assert_eq!(values, vec![0, 1, 2]);
}

#[test]
fn test_insert_and_iter() {
    let mut map = SortedMap::new();
    let mut expected = Vec::new();

    for x in 0..100 {
        assert_eq!(map.iter().cloned().collect::<Vec<_>>(), expected);

        let x = 1000 - x * 2;
        map.insert(x, x);
        expected.insert(0, (x, x));
    }
}

#[test]
fn test_get_and_index() {
    let mut map = SortedMap::new();
    let mut expected = Vec::new();

    for x in 0..100 {
        let x = 1000 - x;
        if x & 1 == 0 {
            map.insert(x, x);
        }
        expected.push(x);
    }

    for mut x in expected {
        if x & 1 == 0 {
            assert_eq!(map.get(&x), Some(&x));
            assert_eq!(map.get_mut(&x), Some(&mut x));
            assert_eq!(map[&x], x);
            assert_eq!(&mut map[&x], &mut x);
        } else {
            assert_eq!(map.get(&x), None);
            assert_eq!(map.get_mut(&x), None);
        }
    }
}

#[test]
fn test_range() {
    let mut map = SortedMap::new();
    map.insert(1, 1);
    map.insert(3, 3);
    map.insert(6, 6);
    map.insert(9, 9);

    let keys = |s: &[(_, _)]| s.into_iter().map(|e| e.0).collect::<Vec<u32>>();

    for start in 0..11 {
        for end in 0..11 {
            if end < start {
                continue;
            }

            let mut expected = vec![1, 3, 6, 9];
            expected.retain(|&x| x >= start && x < end);

            assert_eq!(keys(map.range(start..end)), expected, "range = {}..{}", start, end);
        }
    }
}

#[test]
fn test_offset_keys() {
    let mut map = SortedMap::new();
    map.insert(1, 1);
    map.insert(3, 3);
    map.insert(6, 6);

    map.offset_keys(|k| *k += 1);

    let mut expected = SortedMap::new();
    expected.insert(2, 1);
    expected.insert(4, 3);
    expected.insert(7, 6);

    assert_eq!(map, expected);
}

fn keys(s: SortedMap<u32, u32>) -> Vec<u32> {
    s.into_iter().map(|(k, _)| k).collect::<Vec<u32>>()
}

fn elements(s: SortedMap<u32, u32>) -> Vec<(u32, u32)> {
    s.into_iter().collect::<Vec<(u32, u32)>>()
}

#[test]
fn test_remove_range() {
    let mut map = SortedMap::new();
    map.insert(1, 1);
    map.insert(3, 3);
    map.insert(6, 6);
    map.insert(9, 9);

    for start in 0..11 {
        for end in 0..11 {
            if end < start {
                continue;
            }

            let mut expected = vec![1, 3, 6, 9];
            expected.retain(|&x| x < start || x >= end);

            let mut map = map.clone();
            map.remove_range(start..end);

            assert_eq!(keys(map), expected, "range = {}..{}", start, end);
        }
    }
}

#[test]
fn test_remove() {
    let mut map = SortedMap::new();
    let mut expected = Vec::new();

    for x in 0..10 {
        map.insert(x, x);
        expected.push((x, x));
    }

    for x in 0..10 {
        let mut map = map.clone();
        let mut expected = expected.clone();

        assert_eq!(map.remove(&x), Some(x));
        expected.remove(x as usize);

        assert_eq!(map.iter().cloned().collect::<Vec<_>>(), expected);
    }
}

#[test]
fn test_insert_presorted_non_overlapping() {
    let mut map = SortedMap::new();
    map.insert(2, 0);
    map.insert(8, 0);

    map.insert_presorted(vec![(3, 0), (7, 0)].into_iter());

    let expected = vec![2, 3, 7, 8];
    assert_eq!(keys(map), expected);
}

#[test]
fn test_insert_presorted_first_elem_equal() {
    let mut map = SortedMap::new();
    map.insert(2, 2);
    map.insert(8, 8);

    map.insert_presorted(vec![(2, 0), (7, 7)].into_iter());

    let expected = vec![(2, 0), (7, 7), (8, 8)];
    assert_eq!(elements(map), expected);
}

#[test]
fn test_insert_presorted_last_elem_equal() {
    let mut map = SortedMap::new();
    map.insert(2, 2);
    map.insert(8, 8);

    map.insert_presorted(vec![(3, 3), (8, 0)].into_iter());

    let expected = vec![(2, 2), (3, 3), (8, 0)];
    assert_eq!(elements(map), expected);
}

#[test]
fn test_insert_presorted_shuffle() {
    let mut map = SortedMap::new();
    map.insert(2, 2);
    map.insert(7, 7);

    map.insert_presorted(vec![(1, 1), (3, 3), (8, 8)].into_iter());

    let expected = vec![(1, 1), (2, 2), (3, 3), (7, 7), (8, 8)];
    assert_eq!(elements(map), expected);
}

#[test]
fn test_insert_presorted_at_end() {
    let mut map = SortedMap::new();
    map.insert(1, 1);
    map.insert(2, 2);

    map.insert_presorted(vec![(3, 3), (8, 8)].into_iter());

    let expected = vec![(1, 1), (2, 2), (3, 3), (8, 8)];
    assert_eq!(elements(map), expected);
}
