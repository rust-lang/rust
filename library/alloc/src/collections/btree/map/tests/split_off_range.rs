use super::*;
use std::cmp::{max, min};

#[test]
fn empty() {
    let mut map: BTreeMap<i32, i32> = BTreeMap::new();
    map.split_off_range(..);
    assert!(map.is_empty());
    map.check();
}

// Drop the iterator, where most test cases consume it entirely.
#[test]
fn dropped_keeping_all() {
    let pairs = (0..3).map(|i| (i, i));
    let mut map: BTreeMap<_, _> = pairs.collect();
    map.split_off_range(..0);
    assert!(map.keys().copied().eq(0..3));
    map.check();
}

// Drop the iterator, where most test cases consume it entirely.
#[test]
fn dropped_removing_all() {
    let pairs = (0..3).map(|i| (i, i));
    let mut map: BTreeMap<_, _> = pairs.clone().collect();
    map.split_off_range(..);
    assert!(map.is_empty());
    map.check();
}

#[test]
fn consumed_keeping_all() {
    let pairs = (0..3).map(|i| (i, ()));
    let mut map: BTreeMap<_, _> = pairs.collect();
    assert!(map.split_off_range(..0).eq(&BTreeMap::new()));
    assert!(map.keys().copied().eq(0..3));
    map.check();
}

fn test_size_range<R: RangeBounds<usize>>(
    size: usize,
    height: usize,
    compact: bool,
    keep: usize,
    range: R,
) {
    let mut map: BTreeMap<_, _> = (0..size).map(|i| (i, ())).collect();
    if compact {
        map.compact();
    }
    assert_eq!(map.height(), Some(height));
    let split_off = map.split_off_range(range);
    assert_eq!(split_off.len(), size - keep);
    assert_eq!(map.len(), keep);
    split_off.check();
    map.check();
}

fn test_size_keeping_n(size: usize, height: usize, compact: bool, keep: usize) {
    for doomed_start in 0..keep + 1 {
        test_size_range(size, height, compact, keep, doomed_start..(doomed_start + size - keep));
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
    test_size_range(MIN_INSERTS_HEIGHT_2, 2, false, 12, 0..77);
}

#[test]
fn height_2_min_keeping_12_mid() {
    test_size_range(MIN_INSERTS_HEIGHT_2, 2, false, 12, 6..83);
}

#[test]
fn height_2_min_keeping_12_right() {
    test_size_range(MIN_INSERTS_HEIGHT_2, 2, false, 12, 12..89)
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
        let split_off = map.split_off_range(bounds.clone());
        assert_eq!(split_off.len() + map.len(), size);
        split_off.check();
        map.check();
        assert!(split_off.into_keys().all(|k| bounds.contains(&k)));
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
    catch_unwind(move || drop(map.split_off_range(..))).unwrap_err();
    assert_eq!(DROPS.load(SeqCst), 0x111);
}
