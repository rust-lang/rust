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
fn bitset_clone_from() {
    let mut a: BitSet<usize> = BitSet::new_empty(10);
    a.insert(4);
    a.insert(7);
    a.insert(9);

    let mut b = BitSet::new_empty(2);
    b.clone_from(&a);
    assert_eq!(b.domain_size(), 10);
    assert_eq!(b.iter().collect::<Vec<_>>(), [4, 7, 9]);

    b.clone_from(&BitSet::new_empty(40));
    assert_eq!(b.domain_size(), 40);
    assert_eq!(b.iter().collect::<Vec<_>>(), []);
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
fn chunked_bitset() {
    let mut b0 = ChunkedBitSet::<usize>::new_empty(0);
    let b0b = b0.clone();
    assert_eq!(b0, ChunkedBitSet { domain_size: 0, chunks: Box::new([]), marker: PhantomData });

    // There are no valid insert/remove/contains operations on a 0-domain
    // bitset, but we can test `union`.
    b0.assert_valid();
    assert!(!b0.union(&b0b));
    assert_eq!(b0.chunks(), vec![]);
    assert_eq!(b0.count(), 0);
    b0.assert_valid();

    //-----------------------------------------------------------------------

    let mut b1 = ChunkedBitSet::<usize>::new_empty(1);
    assert_eq!(
        b1,
        ChunkedBitSet { domain_size: 1, chunks: Box::new([Zeros(1)]), marker: PhantomData }
    );

    b1.assert_valid();
    assert!(!b1.contains(0));
    assert_eq!(b1.count(), 0);
    assert!(b1.insert(0));
    assert!(b1.contains(0));
    assert_eq!(b1.count(), 1);
    assert_eq!(b1.chunks(), [Ones(1)]);
    assert!(!b1.insert(0));
    assert!(b1.remove(0));
    assert!(!b1.contains(0));
    assert_eq!(b1.count(), 0);
    assert_eq!(b1.chunks(), [Zeros(1)]);
    b1.assert_valid();

    //-----------------------------------------------------------------------

    let mut b100 = ChunkedBitSet::<usize>::new_filled(100);
    assert_eq!(
        b100,
        ChunkedBitSet { domain_size: 100, chunks: Box::new([Ones(100)]), marker: PhantomData }
    );

    b100.assert_valid();
    for i in 0..100 {
        assert!(b100.contains(i));
    }
    assert_eq!(b100.count(), 100);
    assert!(b100.remove(3));
    assert!(b100.insert(3));
    assert_eq!(b100.chunks(), vec![Ones(100)]);
    assert!(
        b100.remove(20) && b100.remove(30) && b100.remove(40) && b100.remove(99) && b100.insert(30)
    );
    assert_eq!(b100.count(), 97);
    assert!(!b100.contains(20) && b100.contains(30) && !b100.contains(99) && b100.contains(50));
    assert_eq!(
        b100.chunks(),
        vec![Mixed(
            100,
            97,
            #[rustfmt::skip]
            Rc::new([
                0b11111111_11111111_11111110_11111111_11111111_11101111_11111111_11111111,
                0b00000000_00000000_00000000_00000111_11111111_11111111_11111111_11111111,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0,
            ])
        )],
    );
    b100.assert_valid();
    let mut num_removed = 0;
    for i in 0..100 {
        if b100.remove(i) {
            num_removed += 1;
        }
    }
    assert_eq!(num_removed, 97);
    assert_eq!(b100.chunks(), vec![Zeros(100)]);
    b100.assert_valid();

    //-----------------------------------------------------------------------

    let mut b2548 = ChunkedBitSet::<usize>::new_empty(2548);
    assert_eq!(
        b2548,
        ChunkedBitSet {
            domain_size: 2548,
            chunks: Box::new([Zeros(2048), Zeros(500)]),
            marker: PhantomData,
        }
    );

    b2548.assert_valid();
    b2548.insert(14);
    b2548.remove(14);
    assert_eq!(b2548.chunks(), vec![Zeros(2048), Zeros(500)]);
    b2548.insert_all();
    for i in 0..2548 {
        assert!(b2548.contains(i));
    }
    assert_eq!(b2548.count(), 2548);
    assert_eq!(b2548.chunks(), vec![Ones(2048), Ones(500)]);
    b2548.assert_valid();

    //-----------------------------------------------------------------------

    let mut b4096 = ChunkedBitSet::<usize>::new_empty(4096);
    assert_eq!(
        b4096,
        ChunkedBitSet {
            domain_size: 4096,
            chunks: Box::new([Zeros(2048), Zeros(2048)]),
            marker: PhantomData,
        }
    );

    b4096.assert_valid();
    for i in 0..4096 {
        assert!(!b4096.contains(i));
    }
    assert!(b4096.insert(0) && b4096.insert(4095) && !b4096.insert(4095));
    assert!(
        b4096.contains(0) && !b4096.contains(2047) && !b4096.contains(2048) && b4096.contains(4095)
    );
    assert_eq!(
        b4096.chunks(),
        #[rustfmt::skip]
        vec![
            Mixed(2048, 1, Rc::new([
                1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
            ])),
            Mixed(2048, 1, Rc::new([
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x8000_0000_0000_0000
            ])),
        ],
    );
    assert_eq!(b4096.count(), 2);
    b4096.assert_valid();

    //-----------------------------------------------------------------------

    let mut b10000 = ChunkedBitSet::<usize>::new_empty(10000);
    assert_eq!(
        b10000,
        ChunkedBitSet {
            domain_size: 10000,
            chunks: Box::new([Zeros(2048), Zeros(2048), Zeros(2048), Zeros(2048), Zeros(1808),]),
            marker: PhantomData,
        }
    );

    b10000.assert_valid();
    assert!(b10000.insert(3000) && b10000.insert(5000));
    assert_eq!(
        b10000.chunks(),
        #[rustfmt::skip]
        vec![
            Zeros(2048),
            Mixed(2048, 1, Rc::new([
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x0100_0000_0000_0000, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            ])),
            Mixed(2048, 1, Rc::new([
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x0100, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            ])),
            Zeros(2048),
            Zeros(1808),
        ],
    );
    let mut b10000b = ChunkedBitSet::<usize>::new_empty(10000);
    b10000b.clone_from(&b10000);
    assert_eq!(b10000, b10000b);
    for i in 6000..7000 {
        b10000b.insert(i);
    }
    assert_eq!(b10000b.count(), 1002);
    b10000b.assert_valid();
    b10000b.clone_from(&b10000);
    assert_eq!(b10000b.count(), 2);
    for i in 2000..8000 {
        b10000b.insert(i);
    }
    b10000.union(&b10000b);
    assert_eq!(b10000.count(), 6000);
    b10000.union(&b10000b);
    assert_eq!(b10000.count(), 6000);
    b10000.assert_valid();
    b10000b.assert_valid();
}

fn with_elements_chunked(elements: &[usize], domain_size: usize) -> ChunkedBitSet<usize> {
    let mut s = ChunkedBitSet::new_empty(domain_size);
    for &e in elements {
        assert!(s.insert(e));
    }
    s
}

fn with_elements_standard(elements: &[usize], domain_size: usize) -> BitSet<usize> {
    let mut s = BitSet::new_empty(domain_size);
    for &e in elements {
        assert!(s.insert(e));
    }
    s
}

#[test]
fn chunked_bitset_into_bitset_operations() {
    let a = vec![1, 5, 7, 11, 15, 2000, 3000];
    let b = vec![3, 4, 11, 3000, 4000];
    let aub = vec![1, 3, 4, 5, 7, 11, 15, 2000, 3000, 4000];
    let aib = vec![11, 3000];

    let b = with_elements_chunked(&b, 9876);

    let mut union = with_elements_standard(&a, 9876);
    assert!(union.union(&b));
    assert!(!union.union(&b));
    assert!(union.iter().eq(aub.iter().copied()));

    let mut intersection = with_elements_standard(&a, 9876);
    assert!(intersection.intersect(&b));
    assert!(!intersection.intersect(&b));
    assert!(intersection.iter().eq(aib.iter().copied()));
}

#[test]
fn chunked_bitset_iter() {
    fn check_iter(bit: &ChunkedBitSet<usize>, vec: &Vec<usize>) {
        // Test collecting via both `.next()` and `.fold()` calls, to make sure both are correct
        let mut collect_next = Vec::new();
        let mut bit_iter = bit.iter();
        while let Some(item) = bit_iter.next() {
            collect_next.push(item);
        }
        assert_eq!(vec, &collect_next);

        let collect_fold = bit.iter().fold(Vec::new(), |mut v, item| {
            v.push(item);
            v
        });
        assert_eq!(vec, &collect_fold);
    }

    // Empty
    let vec: Vec<usize> = Vec::new();
    let bit = with_elements_chunked(&vec, 9000);
    check_iter(&bit, &vec);

    // Filled
    let n = 10000;
    let vec: Vec<usize> = (0..n).collect();
    let bit = with_elements_chunked(&vec, n);
    check_iter(&bit, &vec);

    // Filled with trailing zeros
    let n = 10000;
    let vec: Vec<usize> = (0..n).collect();
    let bit = with_elements_chunked(&vec, 2 * n);
    check_iter(&bit, &vec);

    // Mixed
    let n = 12345;
    let vec: Vec<usize> = vec![0, 1, 2, 2010, 2047, 2099, 6000, 6002, 6004];
    let bit = with_elements_chunked(&vec, n);
    check_iter(&bit, &vec);
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
