use super::*;

#[test]
fn insert_collapses() {
    let mut set = IntervalSet::<u32>::new(10000);
    set.insert_range(9831..=9837);
    set.insert_range(43..=9830);
    assert_eq!(set.iter_intervals().collect::<Vec<_>>(), [43..9838]);
}

#[test]
fn contains() {
    let mut set = IntervalSet::new(300);
    set.insert(0u32);
    assert!(set.contains(0));
    set.insert_range(0..10);
    assert!(set.contains(9));
    assert!(!set.contains(10));
    set.insert_range(10..11);
    assert!(set.contains(10));
}

#[test]
fn insert() {
    for i in 0..30usize {
        let mut set = IntervalSet::new(300);
        for j in i..30usize {
            set.insert(j);
            for k in i..j {
                assert!(set.contains(k));
            }
        }
    }

    let mut set = IntervalSet::new(300);
    set.insert_range(0..1u32);
    assert!(set.contains(0), "{:?}", set.map);
    assert!(!set.contains(1));
    set.insert_range(1..1);
    assert!(set.contains(0));
    assert!(!set.contains(1));

    let mut set = IntervalSet::new(300);
    set.insert_range(4..5u32);
    set.insert_range(5..10);
    assert_eq!(set.iter().collect::<Vec<_>>(), [4, 5, 6, 7, 8, 9]);
    set.insert_range(3..7);
    assert_eq!(set.iter().collect::<Vec<_>>(), [3, 4, 5, 6, 7, 8, 9]);

    let mut set = IntervalSet::new(300);
    set.insert_range(0..10u32);
    set.insert_range(3..5);
    assert_eq!(set.iter().collect::<Vec<_>>(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);

    let mut set = IntervalSet::new(300);
    set.insert_range(0..10u32);
    set.insert_range(0..3);
    assert_eq!(set.iter().collect::<Vec<_>>(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);

    let mut set = IntervalSet::new(300);
    set.insert_range(0..10u32);
    set.insert_range(0..10);
    assert_eq!(set.iter().collect::<Vec<_>>(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);

    let mut set = IntervalSet::new(300);
    set.insert_range(0..10u32);
    set.insert_range(5..10);
    assert_eq!(set.iter().collect::<Vec<_>>(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);

    let mut set = IntervalSet::new(300);
    set.insert_range(0..10u32);
    set.insert_range(5..13);
    assert_eq!(set.iter().collect::<Vec<_>>(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
}

#[test]
fn insert_range() {
    #[track_caller]
    fn check<R>(range: R)
    where
        R: RangeBounds<usize> + Clone + IntoIterator<Item = usize> + std::fmt::Debug,
    {
        let mut set = IntervalSet::new(300);
        set.insert_range(range.clone());
        for i in set.iter() {
            assert!(range.contains(&i));
        }
        for i in range.clone() {
            assert!(set.contains(i), "A: {} in {:?}, inserted {:?}", i, set, range);
        }
        set.insert_range(range.clone());
        for i in set.iter() {
            assert!(range.contains(&i), "{} in {:?}", i, set);
        }
        for i in range.clone() {
            assert!(set.contains(i), "B: {} in {:?}, inserted {:?}", i, set, range);
        }
    }
    check(10..10);
    check(10..100);
    check(10..30);
    check(0..5);
    check(0..250);
    check(200..250);

    check(10..=10);
    check(10..=100);
    check(10..=30);
    check(0..=5);
    check(0..=250);
    check(200..=250);

    for i in 0..30 {
        for j in i..30 {
            check(i..j);
            check(i..=j);
        }
    }
}

#[test]
fn insert_range_dual() {
    let mut set = IntervalSet::<u32>::new(300);
    set.insert_range(0..3);
    assert_eq!(set.iter().collect::<Vec<_>>(), [0, 1, 2]);
    set.insert_range(5..7);
    assert_eq!(set.iter().collect::<Vec<_>>(), [0, 1, 2, 5, 6]);
    set.insert_range(3..4);
    assert_eq!(set.iter().collect::<Vec<_>>(), [0, 1, 2, 3, 5, 6]);
    set.insert_range(3..5);
    assert_eq!(set.iter().collect::<Vec<_>>(), [0, 1, 2, 3, 4, 5, 6]);
}

#[test]
fn last_set_before_adjacent() {
    let mut set = IntervalSet::<u32>::new(300);
    set.insert_range(0..3);
    set.insert_range(3..5);
    assert_eq!(set.last_set_in(0..3), Some(2));
    assert_eq!(set.last_set_in(0..5), Some(4));
    assert_eq!(set.last_set_in(3..5), Some(4));
    set.insert_range(2..5);
    assert_eq!(set.last_set_in(0..3), Some(2));
    assert_eq!(set.last_set_in(0..5), Some(4));
    assert_eq!(set.last_set_in(3..5), Some(4));
}

#[test]
fn last_set_in() {
    fn easy(set: &IntervalSet<usize>, needle: impl RangeBounds<usize>) -> Option<usize> {
        let mut last_leq = None;
        for e in set.iter() {
            if needle.contains(&e) {
                last_leq = Some(e);
            }
        }
        last_leq
    }

    #[track_caller]
    fn cmp(set: &IntervalSet<usize>, needle: impl RangeBounds<usize> + Clone + std::fmt::Debug) {
        assert_eq!(
            set.last_set_in(needle.clone()),
            easy(set, needle.clone()),
            "{:?} in {:?}",
            needle,
            set
        );
    }
    let mut set = IntervalSet::new(300);
    cmp(&set, 50..=50);
    set.insert(64);
    cmp(&set, 64..=64);
    set.insert(64 - 1);
    cmp(&set, 0..=64 - 1);
    cmp(&set, 0..=5);
    cmp(&set, 10..100);
    set.insert(100);
    cmp(&set, 100..110);
    cmp(&set, 99..100);
    cmp(&set, 99..=100);

    for i in 0..=30 {
        for j in i..=30 {
            for k in 0..30 {
                let mut set = IntervalSet::new(100);
                cmp(&set, ..j);
                cmp(&set, i..);
                cmp(&set, i..j);
                cmp(&set, i..=j);
                set.insert(k);
                cmp(&set, ..j);
                cmp(&set, i..);
                cmp(&set, i..j);
                cmp(&set, i..=j);
            }
        }
    }
}
