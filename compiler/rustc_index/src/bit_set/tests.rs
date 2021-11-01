use super::*;

extern crate test;
use std::hint::black_box;
use test::Bencher;

#[test]
fn test_new_filled() {
    for i in 0..128 {
        let idx_buf = BitSet::new_filled(i);
        let elems: Vec<usize> = idx_buf.iter().collect();
        let expected: Vec<usize> = (0..i).collect();
        assert_eq!(elems, expected);
    }
}

#[test]
fn bitset_iter_works() {
    let mut bitset: BitSet<usize> = BitSet::new_empty(100);
    bitset.insert(1);
    bitset.insert(10);
    bitset.insert(19);
    bitset.insert(62);
    bitset.insert(63);
    bitset.insert(64);
    bitset.insert(65);
    bitset.insert(66);
    bitset.insert(99);
    assert_eq!(bitset.iter().collect::<Vec<_>>(), [1, 10, 19, 62, 63, 64, 65, 66, 99]);
}

#[test]
fn bitset_iter_works_2() {
    let mut bitset: BitSet<usize> = BitSet::new_empty(320);
    bitset.insert(0);
    bitset.insert(127);
    bitset.insert(191);
    bitset.insert(255);
    bitset.insert(319);
    assert_eq!(bitset.iter().collect::<Vec<_>>(), [0, 127, 191, 255, 319]);
}

#[test]
fn union_two_sets() {
    let mut set1: BitSet<usize> = BitSet::new_empty(65);
    let mut set2: BitSet<usize> = BitSet::new_empty(65);
    assert!(set1.insert(3));
    assert!(!set1.insert(3));
    assert!(set2.insert(5));
    assert!(set2.insert(64));
    assert!(set1.union(&set2));
    assert!(!set1.union(&set2));
    assert!(set1.contains(3));
    assert!(!set1.contains(4));
    assert!(set1.contains(5));
    assert!(!set1.contains(63));
    assert!(set1.contains(64));
}

#[test]
fn hybrid_bitset() {
    let mut sparse038: HybridBitSet<usize> = HybridBitSet::new_empty(256);
    assert!(sparse038.is_empty());
    assert!(sparse038.insert(0));
    assert!(sparse038.insert(1));
    assert!(sparse038.insert(8));
    assert!(sparse038.insert(3));
    assert!(!sparse038.insert(3));
    assert!(sparse038.remove(1));
    assert!(!sparse038.is_empty());
    assert_eq!(sparse038.iter().collect::<Vec<_>>(), [0, 3, 8]);

    for i in 0..256 {
        if i == 0 || i == 3 || i == 8 {
            assert!(sparse038.contains(i));
        } else {
            assert!(!sparse038.contains(i));
        }
    }

    let mut sparse01358 = sparse038.clone();
    assert!(sparse01358.insert(1));
    assert!(sparse01358.insert(5));
    assert_eq!(sparse01358.iter().collect::<Vec<_>>(), [0, 1, 3, 5, 8]);

    let mut dense10 = HybridBitSet::new_empty(256);
    for i in 0..10 {
        assert!(dense10.insert(i));
    }
    assert!(!dense10.is_empty());
    assert_eq!(dense10.iter().collect::<Vec<_>>(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);

    let mut dense256 = HybridBitSet::new_empty(256);
    assert!(dense256.is_empty());
    dense256.insert_all();
    assert!(!dense256.is_empty());
    for i in 0..256 {
        assert!(dense256.contains(i));
    }

    assert!(sparse038.superset(&sparse038)); // sparse + sparse (self)
    assert!(sparse01358.superset(&sparse038)); // sparse + sparse
    assert!(dense10.superset(&sparse038)); // dense + sparse
    assert!(dense10.superset(&dense10)); // dense + dense (self)
    assert!(dense256.superset(&dense10)); // dense + dense

    let mut hybrid = sparse038.clone();
    assert!(!sparse01358.union(&hybrid)); // no change
    assert!(hybrid.union(&sparse01358));
    assert!(hybrid.superset(&sparse01358) && sparse01358.superset(&hybrid));
    assert!(!dense256.union(&dense10));

    // dense / sparse where dense superset sparse
    assert!(!dense10.clone().union(&sparse01358));
    assert!(sparse01358.clone().union(&dense10));
    assert!(dense10.clone().intersect(&sparse01358));
    assert!(!sparse01358.clone().intersect(&dense10));
    assert!(dense10.clone().subtract(&sparse01358));
    assert!(sparse01358.clone().subtract(&dense10));

    // dense / sparse where sparse superset dense
    let dense038 = sparse038.to_dense();
    assert!(!sparse01358.clone().union(&dense038));
    assert!(dense038.clone().union(&sparse01358));
    assert!(sparse01358.clone().intersect(&dense038));
    assert!(!dense038.clone().intersect(&sparse01358));
    assert!(sparse01358.clone().subtract(&dense038));
    assert!(dense038.clone().subtract(&sparse01358));

    let mut dense = dense10.clone();
    assert!(dense.union(&dense256));
    assert!(dense.superset(&dense256) && dense256.superset(&dense));
    assert!(hybrid.union(&dense256));
    assert!(hybrid.superset(&dense256) && dense256.superset(&hybrid));

    assert!(!dense10.clone().intersect(&dense256));
    assert!(dense256.clone().intersect(&dense10));
    assert!(dense10.clone().subtract(&dense256));
    assert!(dense256.clone().subtract(&dense10));

    assert_eq!(dense256.iter().count(), 256);
    let mut dense0 = dense256;
    for i in 0..256 {
        assert!(dense0.remove(i));
    }
    assert!(!dense0.remove(0));
    assert!(dense0.is_empty());
}

#[test]
fn grow() {
    let mut set: GrowableBitSet<usize> = GrowableBitSet::with_capacity(65);
    for index in 0..65 {
        assert!(set.insert(index));
        assert!(!set.insert(index));
    }
    set.ensure(128);

    // Check if the bits set before growing are still set
    for index in 0..65 {
        assert!(set.contains(index));
    }

    // Check if the new bits are all un-set
    for index in 65..128 {
        assert!(!set.contains(index));
    }

    // Check that we can set all new bits without running out of bounds
    for index in 65..128 {
        assert!(set.insert(index));
        assert!(!set.insert(index));
    }
}

#[test]
fn matrix_intersection() {
    let mut matrix: BitMatrix<usize, usize> = BitMatrix::new(200, 200);

    // (*) Elements reachable from both 2 and 65.

    matrix.insert(2, 3);
    matrix.insert(2, 6);
    matrix.insert(2, 10); // (*)
    matrix.insert(2, 64); // (*)
    matrix.insert(2, 65);
    matrix.insert(2, 130);
    matrix.insert(2, 160); // (*)

    matrix.insert(64, 133);

    matrix.insert(65, 2);
    matrix.insert(65, 8);
    matrix.insert(65, 10); // (*)
    matrix.insert(65, 64); // (*)
    matrix.insert(65, 68);
    matrix.insert(65, 133);
    matrix.insert(65, 160); // (*)

    let intersection = matrix.intersect_rows(2, 64);
    assert!(intersection.is_empty());

    let intersection = matrix.intersect_rows(2, 65);
    assert_eq!(intersection, &[10, 64, 160]);
}

#[test]
fn matrix_iter() {
    let mut matrix: BitMatrix<usize, usize> = BitMatrix::new(64, 100);
    matrix.insert(3, 22);
    matrix.insert(3, 75);
    matrix.insert(2, 99);
    matrix.insert(4, 0);
    matrix.union_rows(3, 5);
    matrix.insert_all_into_row(6);

    let expected = [99];
    let mut iter = expected.iter();
    for i in matrix.iter(2) {
        let j = *iter.next().unwrap();
        assert_eq!(i, j);
    }
    assert!(iter.next().is_none());

    let expected = [22, 75];
    let mut iter = expected.iter();
    assert_eq!(matrix.count(3), expected.len());
    for i in matrix.iter(3) {
        let j = *iter.next().unwrap();
        assert_eq!(i, j);
    }
    assert!(iter.next().is_none());

    let expected = [0];
    let mut iter = expected.iter();
    assert_eq!(matrix.count(4), expected.len());
    for i in matrix.iter(4) {
        let j = *iter.next().unwrap();
        assert_eq!(i, j);
    }
    assert!(iter.next().is_none());

    let expected = [22, 75];
    let mut iter = expected.iter();
    assert_eq!(matrix.count(5), expected.len());
    for i in matrix.iter(5) {
        let j = *iter.next().unwrap();
        assert_eq!(i, j);
    }
    assert!(iter.next().is_none());

    assert_eq!(matrix.count(6), 100);
    let mut count = 0;
    for (idx, i) in matrix.iter(6).enumerate() {
        assert_eq!(idx, i);
        count += 1;
    }
    assert_eq!(count, 100);

    if let Some(i) = matrix.iter(7).next() {
        panic!("expected no elements in row, but contains element {:?}", i);
    }
}

#[test]
fn sparse_matrix_iter() {
    let mut matrix: SparseBitMatrix<usize, usize> = SparseBitMatrix::new(100);
    matrix.insert(3, 22);
    matrix.insert(3, 75);
    matrix.insert(2, 99);
    matrix.insert(4, 0);
    matrix.union_rows(3, 5);

    let expected = [99];
    let mut iter = expected.iter();
    for i in matrix.iter(2) {
        let j = *iter.next().unwrap();
        assert_eq!(i, j);
    }
    assert!(iter.next().is_none());

    let expected = [22, 75];
    let mut iter = expected.iter();
    for i in matrix.iter(3) {
        let j = *iter.next().unwrap();
        assert_eq!(i, j);
    }
    assert!(iter.next().is_none());

    let expected = [0];
    let mut iter = expected.iter();
    for i in matrix.iter(4) {
        let j = *iter.next().unwrap();
        assert_eq!(i, j);
    }
    assert!(iter.next().is_none());

    let expected = [22, 75];
    let mut iter = expected.iter();
    for i in matrix.iter(5) {
        let j = *iter.next().unwrap();
        assert_eq!(i, j);
    }
    assert!(iter.next().is_none());
}

#[test]
fn sparse_matrix_operations() {
    let mut matrix: SparseBitMatrix<usize, usize> = SparseBitMatrix::new(100);
    matrix.insert(3, 22);
    matrix.insert(3, 75);
    matrix.insert(2, 99);
    matrix.insert(4, 0);

    let mut disjoint: HybridBitSet<usize> = HybridBitSet::new_empty(100);
    disjoint.insert(33);

    let mut superset = HybridBitSet::new_empty(100);
    superset.insert(22);
    superset.insert(75);
    superset.insert(33);

    let mut subset = HybridBitSet::new_empty(100);
    subset.insert(22);

    // SparseBitMatrix::remove
    {
        let mut matrix = matrix.clone();
        matrix.remove(3, 22);
        assert!(!matrix.row(3).unwrap().contains(22));
        matrix.remove(0, 0);
        assert!(matrix.row(0).is_none());
    }

    // SparseBitMatrix::clear
    {
        let mut matrix = matrix.clone();
        matrix.clear(3);
        assert!(!matrix.row(3).unwrap().contains(75));
        matrix.clear(0);
        assert!(matrix.row(0).is_none());
    }

    // SparseBitMatrix::intersect_row
    {
        let mut matrix = matrix.clone();
        assert!(!matrix.intersect_row(3, &superset));
        assert!(matrix.intersect_row(3, &subset));
        matrix.intersect_row(0, &disjoint);
        assert!(matrix.row(0).is_none());
    }

    // SparseBitMatrix::subtract_row
    {
        let mut matrix = matrix.clone();
        assert!(!matrix.subtract_row(3, &disjoint));
        assert!(matrix.subtract_row(3, &subset));
        assert!(matrix.subtract_row(3, &superset));
        matrix.intersect_row(0, &disjoint);
        assert!(matrix.row(0).is_none());
    }

    // SparseBitMatrix::union_row
    {
        let mut matrix = matrix.clone();
        assert!(!matrix.union_row(3, &subset));
        assert!(matrix.union_row(3, &disjoint));
        matrix.union_row(0, &disjoint);
        assert!(matrix.row(0).is_some());
    }
}

#[test]
fn dense_insert_range() {
    #[track_caller]
    fn check<R>(domain: usize, range: R)
    where
        R: RangeBounds<usize> + Clone + IntoIterator<Item = usize> + std::fmt::Debug,
    {
        let mut set = BitSet::new_empty(domain);
        set.insert_range(range.clone());
        for i in set.iter() {
            assert!(range.contains(&i));
        }
        for i in range.clone() {
            assert!(set.contains(i), "{} in {:?}, inserted {:?}", i, set, range);
        }
    }
    check(300, 10..10);
    check(300, WORD_BITS..WORD_BITS * 2);
    check(300, WORD_BITS - 1..WORD_BITS * 2);
    check(300, WORD_BITS - 1..WORD_BITS);
    check(300, 10..100);
    check(300, 10..30);
    check(300, 0..5);
    check(300, 0..250);
    check(300, 200..250);

    check(300, 10..=10);
    check(300, WORD_BITS..=WORD_BITS * 2);
    check(300, WORD_BITS - 1..=WORD_BITS * 2);
    check(300, WORD_BITS - 1..=WORD_BITS);
    check(300, 10..=100);
    check(300, 10..=30);
    check(300, 0..=5);
    check(300, 0..=250);
    check(300, 200..=250);

    for i in 0..WORD_BITS * 2 {
        for j in i..WORD_BITS * 2 {
            check(WORD_BITS * 2, i..j);
            check(WORD_BITS * 2, i..=j);
            check(300, i..j);
            check(300, i..=j);
        }
    }
}

#[test]
fn dense_last_set_before() {
    fn easy(set: &BitSet<usize>, needle: impl RangeBounds<usize>) -> Option<usize> {
        let mut last_leq = None;
        for e in set.iter() {
            if needle.contains(&e) {
                last_leq = Some(e);
            }
        }
        last_leq
    }

    #[track_caller]
    fn cmp(set: &BitSet<usize>, needle: impl RangeBounds<usize> + Clone + std::fmt::Debug) {
        assert_eq!(
            set.last_set_in(needle.clone()),
            easy(set, needle.clone()),
            "{:?} in {:?}",
            needle,
            set
        );
    }
    let mut set = BitSet::new_empty(300);
    cmp(&set, 50..=50);
    set.insert(WORD_BITS);
    cmp(&set, WORD_BITS..=WORD_BITS);
    set.insert(WORD_BITS - 1);
    cmp(&set, 0..=WORD_BITS - 1);
    cmp(&set, 0..=5);
    cmp(&set, 10..100);
    set.insert(100);
    cmp(&set, 100..110);
    cmp(&set, 99..100);
    cmp(&set, 99..=100);

    for i in 0..=WORD_BITS * 2 {
        for j in i..=WORD_BITS * 2 {
            for k in 0..WORD_BITS * 2 {
                let mut set = BitSet::new_empty(300);
                cmp(&set, i..j);
                cmp(&set, i..=j);
                set.insert(k);
                cmp(&set, i..j);
                cmp(&set, i..=j);
            }
        }
    }
}

/// Merge dense hybrid set into empty sparse hybrid set.
#[bench]
fn union_hybrid_sparse_empty_to_dense(b: &mut Bencher) {
    let mut pre_dense: HybridBitSet<usize> = HybridBitSet::new_empty(256);
    for i in 0..10 {
        assert!(pre_dense.insert(i));
    }
    let pre_sparse: HybridBitSet<usize> = HybridBitSet::new_empty(256);
    b.iter(|| {
        let dense = pre_dense.clone();
        let mut sparse = pre_sparse.clone();
        sparse.union(&dense);
    })
}

/// Merge dense hybrid set into full hybrid set with same indices.
#[bench]
fn union_hybrid_sparse_full_to_dense(b: &mut Bencher) {
    let mut pre_dense: HybridBitSet<usize> = HybridBitSet::new_empty(256);
    for i in 0..10 {
        assert!(pre_dense.insert(i));
    }
    let mut pre_sparse: HybridBitSet<usize> = HybridBitSet::new_empty(256);
    for i in 0..SPARSE_MAX {
        assert!(pre_sparse.insert(i));
    }
    b.iter(|| {
        let dense = pre_dense.clone();
        let mut sparse = pre_sparse.clone();
        sparse.union(&dense);
    })
}

/// Merge dense hybrid set into full hybrid set with indices over the whole domain.
#[bench]
fn union_hybrid_sparse_domain_to_dense(b: &mut Bencher) {
    let mut pre_dense: HybridBitSet<usize> = HybridBitSet::new_empty(SPARSE_MAX * 64);
    for i in 0..10 {
        assert!(pre_dense.insert(i));
    }
    let mut pre_sparse: HybridBitSet<usize> = HybridBitSet::new_empty(SPARSE_MAX * 64);
    for i in 0..SPARSE_MAX {
        assert!(pre_sparse.insert(i * 64));
    }
    b.iter(|| {
        let dense = pre_dense.clone();
        let mut sparse = pre_sparse.clone();
        sparse.union(&dense);
    })
}

/// Merge dense hybrid set into empty hybrid set where the domain is very small.
#[bench]
fn union_hybrid_sparse_empty_small_domain(b: &mut Bencher) {
    let mut pre_dense: HybridBitSet<usize> = HybridBitSet::new_empty(SPARSE_MAX);
    for i in 0..SPARSE_MAX {
        assert!(pre_dense.insert(i));
    }
    let pre_sparse: HybridBitSet<usize> = HybridBitSet::new_empty(SPARSE_MAX);
    b.iter(|| {
        let dense = pre_dense.clone();
        let mut sparse = pre_sparse.clone();
        sparse.union(&dense);
    })
}

/// Merge dense hybrid set into full hybrid set where the domain is very small.
#[bench]
fn union_hybrid_sparse_full_small_domain(b: &mut Bencher) {
    let mut pre_dense: HybridBitSet<usize> = HybridBitSet::new_empty(SPARSE_MAX);
    for i in 0..SPARSE_MAX {
        assert!(pre_dense.insert(i));
    }
    let mut pre_sparse: HybridBitSet<usize> = HybridBitSet::new_empty(SPARSE_MAX);
    for i in 0..SPARSE_MAX {
        assert!(pre_sparse.insert(i));
    }
    b.iter(|| {
        let dense = pre_dense.clone();
        let mut sparse = pre_sparse.clone();
        sparse.union(&dense);
    })
}

#[bench]
fn bench_insert(b: &mut Bencher) {
    let mut bs = BitSet::new_filled(99999usize);
    b.iter(|| {
        black_box(bs.insert(black_box(100u32)));
    });
}

#[bench]
fn bench_remove(b: &mut Bencher) {
    let mut bs = BitSet::new_filled(99999usize);
    b.iter(|| {
        black_box(bs.remove(black_box(100u32)));
    });
}

#[bench]
fn bench_iter(b: &mut Bencher) {
    let bs = BitSet::new_filled(99999usize);
    b.iter(|| {
        bs.iter().map(|b: usize| black_box(b)).for_each(drop);
    });
}

#[bench]
fn bench_intersect(b: &mut Bencher) {
    let mut ba: BitSet<u32> = BitSet::new_filled(99999usize);
    let bb = BitSet::new_filled(99999usize);
    b.iter(|| {
        ba.intersect(black_box(&bb));
    });
}
