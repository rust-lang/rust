use super::*;
use std::cmp::{max, min};

#[test]
fn empty() {
    let mut map: BTreeMap<i32, i32> = BTreeMap::new();
    map.drain(..);
    assert!(map.is_empty());
    map.check();
}

// Drop the iterator, where most test cases consume it entirely.
#[test]
fn dropped_keeping_all() {
    let pairs = (0..3).map(|i| (i, i));
    let mut map: BTreeMap<_, _> = pairs.collect();
    map.drain(..0);
    assert!(map.keys().copied().eq(0..3));
    map.check();
}

// Drop the iterator, where most test cases consume it entirely.
#[test]
fn dropped_removing_all() {
    let pairs = (0..3).map(|i| (i, i));
    let mut map: BTreeMap<_, _> = pairs.clone().collect();
    map.drain(..);
    assert!(map.is_empty());
    map.check();
}

#[test]
fn consumed_keeping_all() {
    let pairs = (0..3).map(|i| (i, ()));
    let mut map: BTreeMap<_, _> = pairs.collect();
    assert!(map.drain(..0).eq(iter::empty()));
    assert!(map.keys().copied().eq(0..3));
    map.check();
}

#[test]
fn range_small() {
    fn range_keys(size: i32, range: impl RangeBounds<i32>) -> Vec<i32> {
        let mut map: BTreeMap<_, _> = (1..=size).map(|i| (i, ())).collect();
        map.drain(range).map(|kv| kv.0).collect()
    }

    let size = 4;
    let all: Vec<_> = (1..=size).collect();
    let (first, last) = (vec![all[0]], vec![all[size as usize - 1]]);

    assert_eq!(range_keys(size, (Excluded(0), Excluded(size + 1))), all);
    assert_eq!(range_keys(size, (Excluded(0), Included(size + 1))), all);
    assert_eq!(range_keys(size, (Excluded(0), Included(size))), all);
    assert_eq!(range_keys(size, (Excluded(0), Unbounded)), all);
    assert_eq!(range_keys(size, (Included(0), Excluded(size + 1))), all);
    assert_eq!(range_keys(size, (Included(0), Included(size + 1))), all);
    assert_eq!(range_keys(size, (Included(0), Included(size))), all);
    assert_eq!(range_keys(size, (Included(0), Unbounded)), all);
    assert_eq!(range_keys(size, (Included(1), Excluded(size + 1))), all);
    assert_eq!(range_keys(size, (Included(1), Included(size + 1))), all);
    assert_eq!(range_keys(size, (Included(1), Included(size))), all);
    assert_eq!(range_keys(size, (Included(1), Unbounded)), all);
    assert_eq!(range_keys(size, (Unbounded, Excluded(size + 1))), all);
    assert_eq!(range_keys(size, (Unbounded, Included(size + 1))), all);
    assert_eq!(range_keys(size, (Unbounded, Included(size))), all);
    assert_eq!(range_keys(size, ..), all);

    assert_eq!(range_keys(size, (Excluded(0), Excluded(1))), vec![]);
    assert_eq!(range_keys(size, (Excluded(0), Included(0))), vec![]);
    assert_eq!(range_keys(size, (Included(0), Included(0))), vec![]);
    assert_eq!(range_keys(size, (Included(0), Excluded(1))), vec![]);
    assert_eq!(range_keys(size, (Unbounded, Excluded(1))), vec![]);
    assert_eq!(range_keys(size, (Unbounded, Included(0))), vec![]);
    assert_eq!(range_keys(size, (Excluded(0), Excluded(2))), first);
    assert_eq!(range_keys(size, (Excluded(0), Included(1))), first);
    assert_eq!(range_keys(size, (Included(0), Excluded(2))), first);
    assert_eq!(range_keys(size, (Included(0), Included(1))), first);
    assert_eq!(range_keys(size, (Included(1), Excluded(2))), first);
    assert_eq!(range_keys(size, (Included(1), Included(1))), first);
    assert_eq!(range_keys(size, (Unbounded, Excluded(2))), first);
    assert_eq!(range_keys(size, (Unbounded, Included(1))), first);
    assert_eq!(range_keys(size, (Excluded(size - 1), Excluded(size + 1))), last);
    assert_eq!(range_keys(size, (Excluded(size - 1), Included(size + 1))), last);
    assert_eq!(range_keys(size, (Excluded(size - 1), Included(size))), last);
    assert_eq!(range_keys(size, (Excluded(size - 1), Unbounded)), last);
    assert_eq!(range_keys(size, (Included(size), Excluded(size + 1))), last);
    assert_eq!(range_keys(size, (Included(size), Included(size + 1))), last);
    assert_eq!(range_keys(size, (Included(size), Included(size))), last);
    assert_eq!(range_keys(size, (Included(size), Unbounded)), last);
    assert_eq!(range_keys(size, (Excluded(size), Excluded(size + 1))), vec![]);
    assert_eq!(range_keys(size, (Excluded(size), Included(size))), vec![]);
    assert_eq!(range_keys(size, (Excluded(size), Unbounded)), vec![]);
    assert_eq!(range_keys(size, (Included(size + 1), Excluded(size + 1))), vec![]);
    assert_eq!(range_keys(size, (Included(size + 1), Included(size + 1))), vec![]);
    assert_eq!(range_keys(size, (Included(size + 1), Unbounded)), vec![]);
}

fn test_size_range<R: Debug + RangeBounds<usize>>(
    size: usize,
    height: usize,
    compact: bool,
    range: R,
    keep: usize,
) {
    //println!("  range {:?}", range);
    let mut map: BTreeMap<_, _> = (0..size).map(|i| (i, ())).collect();
    if compact {
        map.compact();
    }
    assert_eq!(map.height(), Some(height));
    assert_eq!(map.drain(range).count(), size - keep);
    assert_eq!(map.len(), keep);
    map.check();
}

// Example of a way to debug these test cases.
#[cfg(not(miri))] // Miri is too slow
#[ignore]
#[test]
fn dbg() {
    let size = 181;
    let range = 7..98;
    let mut map: BTreeMap<_, _> = (0..size).map(|i| (i, ())).collect();
    let mut root = map.root.take().unwrap();
    println!("in: {}\n", root.reborrow().dump_keys());
    let drained = root.split_off_range(range);
    println!("kept: {}\n", root.reborrow().dump_keys());
    println!("drained: {}\n", drained.reborrow().dump_keys());
}

fn test_size_keeping_n(size: usize, height: usize, compact: bool, keep: usize) {
    for doomed_start in 0..keep + 1 {
        test_size_range(size, height, compact, doomed_start..(doomed_start + size - keep), keep);
    }
}

fn test_size_all(size: usize, height: usize, compact: bool) {
    for keep in 0..size + 1 {
        test_size_keeping_n(size, height, compact, keep)
    }
}

fn test_size_some(size: usize, height: usize, compact: bool) {
    test_size_keeping_n(size, height, compact, 0);
    test_size_keeping_n(size, height, compact, 1);
    test_size_keeping_n(size, height, compact, 2);
    test_size_keeping_n(size, height, compact, size / 4);
    test_size_keeping_n(size, height, compact, size / 2);
    test_size_keeping_n(size, height, compact, size - 2);
    test_size_keeping_n(size, height, compact, size - 1);
}

#[test]
fn height_0_underfull_all() {
    test_size_all(3, 0, false)
}

#[test]
fn height_0_max_some() {
    test_size_some(NODE_CAPACITY, 0, false)
}

#[cfg(not(miri))] // Miri is too slow
#[test]
fn height_0_max_all() {
    test_size_all(NODE_CAPACITY, 0, false)
}

#[test]
fn height_1_min_keeping_0() {
    test_size_keeping_n(MIN_INSERTS_HEIGHT_1, 1, false, 0)
}

#[test]
fn height_1_min_keeping_1() {
    test_size_keeping_n(MIN_INSERTS_HEIGHT_1, 1, false, 1)
}

#[test]
fn height_1_min_keeping_2() {
    test_size_keeping_n(MIN_INSERTS_HEIGHT_1, 1, false, 2)
}

#[test]
fn height_1_min_keeping_7() {
    test_size_keeping_n(MIN_INSERTS_HEIGHT_1, 1, false, 7)
}

#[cfg(not(miri))] // Miri is too slow
#[test]
fn height_1_min_all() {
    test_size_all(MIN_INSERTS_HEIGHT_1, 1, false)
}

#[cfg(not(miri))] // Miri is too slow
#[test]
fn height_1_more_all() {
    for size in MIN_INSERTS_HEIGHT_1 + 1..MIN_INSERTS_HEIGHT_2 {
        test_size_all(size, 1, false)
    }
}

#[test]
fn height_2_min_keeping_0() {
    test_size_keeping_n(MIN_INSERTS_HEIGHT_2, 2, false, 0)
}

#[test]
fn height_2_min_keeping_1() {
    test_size_keeping_n(MIN_INSERTS_HEIGHT_2, 2, false, 1)
}

#[test]
fn height_2_min_keeping_12_left() {
    test_size_range(MIN_INSERTS_HEIGHT_2, 2, false, 0..77, 12);
}

#[test]
fn height_2_min_keeping_12_mid() {
    test_size_range(MIN_INSERTS_HEIGHT_2, 2, false, 6..83, 12);
}

#[test]
fn height_2_min_keeping_12_right() {
    test_size_range(MIN_INSERTS_HEIGHT_2, 2, false, 12..89, 12)
}

#[cfg(not(miri))] // Miri is too slow
#[test]
fn height_2_min_all() {
    test_size_all(MIN_INSERTS_HEIGHT_2, 2, false)
}

#[cfg(not(miri))] // Miri is too slow
#[ignore]
#[test]
fn height_2_more_some() {
    for size in MIN_INSERTS_HEIGHT_2 + 1..MIN_INSERTS_HEIGHT_3 {
        println!("size {}", size);
        test_size_some(size, 2, false)
    }
}

// Simplest case of `fix_opposite_borders` encountering unmergeable
// internal children of which one ends up underfull.
#[test]
fn size_180() {
    test_size_range(180, 2, false, 36..127, 89)
}

// Simplest case of `fix_opposite_borders` encountering unmergeable
// internal children of which one is empty (and the other full).
#[test]
fn size_181_zero_vs_full() {
    test_size_range(181, 2, false, 7..98, 90)
}

#[cfg(not(miri))] // Miri is too slow
#[test]
fn height_3_min_some() {
    test_size_some(MIN_INSERTS_HEIGHT_3, 3, false)
}

#[cfg(not(miri))] // Miri is too slow
#[ignore]
#[test]
fn height_3_min_all() {
    test_size_all(MIN_INSERTS_HEIGHT_3, 3, false)
}

#[cfg(not(miri))] // Miri is too slow
#[ignore]
#[test]
fn height_4_min_some() {
    test_size_some(MIN_INSERTS_HEIGHT_4, 4, false)
}

#[test]
fn size_143_compact_keeping_1() {
    test_size_keeping_n(143, 1, true, 1)
}

#[cfg(not(miri))] // Miri is too slow
#[test]
fn size_143_compact_all() {
    test_size_all(143, 1, true)
}

#[test]
fn size_144_compact_keeping_1() {
    test_size_keeping_n(144, 2, true, 1)
}

#[cfg(not(miri))] // Miri is too slow
#[test]
fn size_144_compact_all() {
    test_size_all(144, 2, true)
}

#[cfg(not(miri))] // Miri is too slow
#[test]
fn size_1727_compact_some() {
    test_size_some(1727, 2, true)
}

#[cfg(not(miri))] // Miri is too slow
#[test]
fn size_1728_compact_some() {
    test_size_some(1728, 3, true)
}

#[cfg(not(miri))] // Miri is too slow
#[ignore]
#[test]
fn size_20735_compact_some() {
    test_size_some(20735, 3, true)
}

#[cfg(not(miri))] // Miri is too slow
#[ignore]
#[test]
fn size_20736_compact_some() {
    test_size_some(20736, 4, true)
}

#[cfg(not(miri))] // Miri is too slow
#[test]
fn sub_size_143_compact_some() {
    for size in NODE_CAPACITY + 1..143 {
        test_size_some(size, 1, true)
    }
}

#[cfg(not(miri))] // Miri is too slow
#[ignore]
#[test]
fn sub_size_1727_compact_some() {
    for size in (144 + 1..1727).step_by(10) {
        test_size_some(size, 2, true)
    }
}

#[test]
fn random_1() {
    let mut rng = DeterministicRng::new();
    for _ in 0..if cfg!(miri) { 1 } else { 140 } {
        let size = rng.next() as usize % 1024;
        let mut map: BTreeMap<_, ()> = BTreeMap::new();
        for _ in 0..size {
            map.insert(rng.next(), ());
        }
        assert_eq!(map.len(), size);
        let (x, y) = (rng.next(), rng.next());
        let bounds = min(x, y)..max(x, y);
        let mut drained = map.drain(bounds.clone());
        assert_eq!(drained.len() + map.len(), size);
        map.check();
        assert!(drained.all(|(k, _)| bounds.contains(&k)));
        assert!(!map.into_keys().any(|k| bounds.contains(&k)));
    }
}

#[test]
fn drop_panic_leak() {
    static DROPS: AtomicUsize = AtomicUsize::new(0);

    struct D(u8);
    impl Drop for D {
        fn drop(&mut self) {
            DROPS.fetch_add(1 << self.0, SeqCst);
            if self.0 == 4 {
                panic!("panic in `drop`");
            }
        }
    }

    let mut map = (0..3).map(|i| (i * 4, D(i * 4))).collect::<BTreeMap<_, _>>();
    catch_unwind(move || drop(map.drain(..))).unwrap_err();
    assert_eq!(DROPS.load(SeqCst), 0x111);
}
