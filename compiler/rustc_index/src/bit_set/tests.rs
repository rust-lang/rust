use std::collections::BTreeSet;
use std::hash::{BuildHasher, BuildHasherDefault, DefaultHasher};
use std::hint::black_box;
use std::ops::{Range, RangeBounds, RangeInclusive};

use rustc_serialize::{Decodable, Decoder, Encodable, Encoder};
use test::Bencher;

use super::*;
use crate::IndexVec;

extern crate test;

/// A very simple pseudo random generator using linear xorshift.
///
/// [See Wikipedia](https://en.wikipedia.org/wiki/Xorshift). This has 64-bit state and a period
/// of `2^64 - 1`.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed)
    }

    fn next(&mut self) -> usize {
        self.0 ^= self.0 << 7;
        self.0 ^= self.0 >> 9;
        self.0 as usize
    }

    fn next_bool(&mut self) -> bool {
        self.next() % 2 == 0
    }

    /// Sample a range, a subset of `0..=max`.
    ///
    /// The purpose of this method is to make edge cases such as `0..=max` more common.
    fn sample_range(&mut self, max: usize) -> RangeInclusive<usize> {
        let start = match self.next() % 3 {
            0 => 0,
            1 => max,
            2 => self.next() % (max + 1),
            _ => unreachable!(),
        };
        let end = match self.next() % 3 {
            0 => 0,
            1 => max,
            2 => self.next() % (max + 1),
            _ => unreachable!(),
        };
        RangeInclusive::new(start, end)
    }
}

#[derive(Default)]
struct EncoderLittleEndian {
    bytes: Vec<u8>,
}

impl Encoder for EncoderLittleEndian {
    fn emit_usize(&mut self, v: usize) {
        self.bytes.extend(v.to_le_bytes());
    }
    fn emit_u8(&mut self, v: u8) {
        self.bytes.extend(v.to_le_bytes());
    }
    fn emit_u16(&mut self, v: u16) {
        self.bytes.extend(v.to_le_bytes());
    }
    fn emit_u32(&mut self, v: u32) {
        self.bytes.extend(v.to_le_bytes());
    }
    fn emit_u64(&mut self, v: u64) {
        self.bytes.extend(v.to_le_bytes());
    }
    fn emit_u128(&mut self, v: u128) {
        self.bytes.extend(v.to_le_bytes());
    }
    fn emit_isize(&mut self, v: isize) {
        self.bytes.extend(v.to_le_bytes());
    }
    fn emit_i8(&mut self, v: i8) {
        self.bytes.extend(v.to_le_bytes());
    }
    fn emit_i16(&mut self, v: i16) {
        self.bytes.extend(v.to_le_bytes());
    }
    fn emit_i32(&mut self, v: i32) {
        self.bytes.extend(v.to_le_bytes());
    }
    fn emit_i64(&mut self, v: i64) {
        self.bytes.extend(v.to_le_bytes());
    }
    fn emit_i128(&mut self, v: i128) {
        self.bytes.extend(v.to_le_bytes());
    }
    fn emit_raw_bytes(&mut self, v: &[u8]) {
        self.bytes.extend(v);
    }
}

struct DecoderLittleEndian<'a> {
    bytes: &'a [u8],
    /// Remember the original `bytes.len()` so we can calculate how many bytes we've read.
    original_len: usize,
}

impl<'a> DecoderLittleEndian<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, original_len: bytes.len() }
    }
}

impl<'a> Decoder for DecoderLittleEndian<'a> {
    fn read_usize(&mut self) -> usize {
        let (int_bytes, rest) = self.bytes.split_at(size_of::<usize>());
        self.bytes = rest;
        usize::from_le_bytes(int_bytes.try_into().unwrap())
    }
    fn read_u128(&mut self) -> u128 {
        let (int_bytes, rest) = self.bytes.split_at(size_of::<u128>());
        self.bytes = rest;
        u128::from_le_bytes(int_bytes.try_into().unwrap())
    }
    fn read_u64(&mut self) -> u64 {
        let (int_bytes, rest) = self.bytes.split_at(size_of::<u64>());
        self.bytes = rest;
        u64::from_le_bytes(int_bytes.try_into().unwrap())
    }
    fn read_u32(&mut self) -> u32 {
        let (int_bytes, rest) = self.bytes.split_at(size_of::<u32>());
        self.bytes = rest;
        u32::from_le_bytes(int_bytes.try_into().unwrap())
    }
    fn read_u16(&mut self) -> u16 {
        let (int_bytes, rest) = self.bytes.split_at(size_of::<u16>());
        self.bytes = rest;
        u16::from_le_bytes(int_bytes.try_into().unwrap())
    }
    fn read_u8(&mut self) -> u8 {
        let (int_bytes, rest) = self.bytes.split_at(size_of::<u8>());
        self.bytes = rest;
        u8::from_le_bytes(int_bytes.try_into().unwrap())
    }
    fn read_isize(&mut self) -> isize {
        let (int_bytes, rest) = self.bytes.split_at(size_of::<isize>());
        self.bytes = rest;
        isize::from_le_bytes(int_bytes.try_into().unwrap())
    }
    fn read_i128(&mut self) -> i128 {
        let (int_bytes, rest) = self.bytes.split_at(size_of::<i128>());
        self.bytes = rest;
        i128::from_le_bytes(int_bytes.try_into().unwrap())
    }
    fn read_i64(&mut self) -> i64 {
        let (int_bytes, rest) = self.bytes.split_at(size_of::<i64>());
        self.bytes = rest;
        i64::from_le_bytes(int_bytes.try_into().unwrap())
    }
    fn read_i32(&mut self) -> i32 {
        let (int_bytes, rest) = self.bytes.split_at(size_of::<i32>());
        self.bytes = rest;
        i32::from_le_bytes(int_bytes.try_into().unwrap())
    }
    fn read_i16(&mut self) -> i16 {
        let (int_bytes, rest) = self.bytes.split_at(size_of::<i16>());
        self.bytes = rest;
        i16::from_le_bytes(int_bytes.try_into().unwrap())
    }
    fn read_raw_bytes(&mut self, len: usize) -> &[u8] {
        let (bytes, rest) = self.bytes.split_at(len);
        self.bytes = rest;
        bytes
    }
    fn peek_byte(&self) -> u8 {
        self.bytes[0]
    }
    fn position(&self) -> usize {
        self.original_len - self.bytes.len()
    }
}

fn test_with_domain_size(domain_size: usize) {
    const TEST_ITERATIONS: u32 = 512;

    let mut set_1 = DenseBitSet::<usize>::new_empty(domain_size);
    let mut set_1_reference = IndexVec::<usize, bool>::from_elem_n(false, domain_size);
    let mut set_2 = DenseBitSet::<usize>::new_empty(domain_size);
    let mut set_2_reference = IndexVec::<usize, bool>::from_elem_n(false, domain_size);

    let hasher = BuildHasherDefault::<DefaultHasher>::new();

    let mut encoder = EncoderLittleEndian::default();

    let mut rng = Rng::new(42);

    for _ in 0..TEST_ITERATIONS {
        // Make a random operation.
        match rng.next() % 100 {
            0..20 => {
                // Insert in one of the sets.
                if domain_size == 0 {
                    continue;
                }
                let elem = rng.next() % domain_size;
                // Choose set to insert into.
                if rng.next_bool() {
                    assert_eq!(!set_1.contains(elem), set_1.insert(elem));
                    set_1_reference[elem] = true;
                } else {
                    assert_eq!(!set_2.contains(elem), set_2.insert(elem));
                    set_2_reference[elem] = true;
                }
            }
            20..40 => {
                // Insert a range in one of the sets.
                if domain_size == 0 {
                    continue;
                }

                let range = rng.sample_range(domain_size - 1);
                // Choose set to insert into.
                if rng.next_bool() {
                    set_1.insert_range_inclusive(range.clone());
                    for i in range {
                        set_1_reference[i] = true;
                    }
                } else {
                    set_2.insert_range_inclusive(range.clone());
                    for i in range {
                        set_2_reference[i] = true;
                    }
                }
            }
            40..50 => {
                // Test insert_all().
                if rng.next_bool() {
                    set_1.insert_all(domain_size);
                    for x in set_1_reference.iter_mut() {
                        *x = true;
                    }
                } else {
                    set_2.insert_all(domain_size);
                    for x in set_2_reference.iter_mut() {
                        *x = true;
                    }
                }
            }
            50..70 => {
                // Remove from one of the sets.
                if domain_size == 0 {
                    continue;
                }
                let elem = rng.next() % domain_size;
                // Choose set to remove into.
                if rng.next_bool() {
                    assert_eq!(set_1.contains(elem), set_1.remove(elem),);
                    set_1_reference[elem] = false;
                } else {
                    assert_eq!(set_2.contains(elem), set_2.remove(elem),);
                    set_2_reference[elem] = false;
                }
            }
            70..76 => {
                // Union
                let old_set_1 = set_1.clone();
                let changed = set_1.union(&set_2);
                assert_eq!(changed, old_set_1 != set_1);

                // Adjust the reference sets.
                for (x, val) in set_2_reference.iter_enumerated() {
                    set_1_reference[x] |= val;
                }
            }
            76..82 => {
                // Intersection
                let old_set_1 = set_1.clone();
                let changed = set_1.intersect(&set_2);
                assert_eq!(changed, old_set_1 != set_1);

                // Adjust the reference sets.
                for (x, val) in set_2_reference.iter_enumerated() {
                    set_1_reference[x] &= val;
                }
            }
            82..88 => {
                // Subtraction
                let old_set_1 = set_1.clone();
                let changed = set_1.subtract(&set_2);
                assert_eq!(changed, old_set_1 != set_1);

                // Adjust the reference sets.
                for (x, val) in set_2_reference.iter_enumerated() {
                    set_1_reference[x] &= !val;
                }
            }
            88..94 => {
                // Union_not
                set_1.union_not(&set_2, domain_size);

                // Adjust the reference sets.
                for (x, val) in set_2_reference.iter_enumerated() {
                    set_1_reference[x] |= !val;
                }
            }
            94..97 => {
                // Clear
                if rng.next_bool() {
                    set_1.clear();
                    for x in set_1_reference.iter_mut() {
                        *x = false;
                    }
                } else {
                    set_2.clear();
                    for x in set_2_reference.iter_mut() {
                        *x = false;
                    }
                }
            }
            97..100 => {
                // Test new_filled().
                if rng.next_bool() {
                    set_1 = DenseBitSet::new_filled(domain_size);
                    for x in set_1_reference.iter_mut() {
                        *x = true;
                    }
                } else {
                    set_2 = DenseBitSet::new_filled(domain_size);
                    for x in set_2_reference.iter_mut() {
                        *x = true;
                    }
                }
            }
            _ => unreachable!(),
        }

        // Check the contains function.
        for i in 0..domain_size {
            assert_eq!(set_1.contains(i), set_1_reference[i]);
            assert_eq!(set_2.contains(i), set_2_reference[i]);
        }

        // Check iter function.
        assert!(
            set_1.iter().eq(set_1_reference.iter_enumerated().filter(|&(_, &v)| v).map(|(x, _)| x))
        );
        assert!(
            set_2.iter().eq(set_2_reference.iter_enumerated().filter(|&(_, &v)| v).map(|(x, _)| x))
        );

        // Check the superset relation.
        assert_eq!(set_1.superset(&set_2), set_2.iter().all(|x| set_1.contains(x)));

        // Check the `==` operator.
        assert_eq!(set_1 == set_2, set_1_reference == set_2_reference);

        // Check the `hash()` function.
        // If the `set_1` and `set_2` are equal, then their hashes must also be equal.
        if set_1 == set_2 {
            assert_eq!(hasher.hash_one(&set_1), hasher.hash_one(&set_2));
        }

        // Check the count function.
        assert_eq!(set_1.count(), set_1_reference.iter().filter(|&&x| x).count());
        assert_eq!(set_2.count(), set_2_reference.iter().filter(|&&x| x).count());

        // Check `only_one_elem()`.
        if let Some(elem) = set_1.only_one_elem() {
            assert_eq!(set_1.count(), 1);
            assert_eq!(elem, set_1.iter().next().unwrap());
        } else {
            assert_ne!(set_1.count(), 1);
        }

        // Check `last_set_in()`.
        if domain_size > 0 {
            let range = rng.sample_range(domain_size - 1);
            assert_eq!(
                set_1.last_set_in(range.clone()),
                range.clone().filter(|&i| set_1.contains(i)).last()
            );
            assert_eq!(
                set_2.last_set_in(range.clone()),
                range.filter(|&i| set_2.contains(i)).last()
            );
        }

        // Check `Encodable` and `Decodable` implementations.
        if rng.next() as u32 % TEST_ITERATIONS < 128 {
            set_1.encode(&mut encoder);

            let mut decoder = DecoderLittleEndian::new(&encoder.bytes);
            let decoded = DenseBitSet::<usize>::decode(&mut decoder);
            assert_eq!(
                decoder.position(),
                encoder.bytes.len(),
                "All bytes must be read when decoding."
            );

            assert_eq!(set_1, decoded);

            encoder.bytes.clear();
        }
    }
}

fn test_relations_with_chunked_set(domain_size: usize) {
    const TEST_ITERATIONS: u32 = 64;

    let mut dense_set = DenseBitSet::<usize>::new_empty(domain_size);
    let mut chunked_set = ChunkedBitSet::new_empty(domain_size);

    let mut rng = Rng::new(42);

    for _ in 0..TEST_ITERATIONS {
        // Make a random operation.
        match rng.next() % 10 {
            0..3 => {
                // Insert in one of the sets.
                if domain_size == 0 {
                    continue;
                }
                let elem = rng.next() % domain_size;
                // Choose set to insert into.
                if rng.next_bool() {
                    dense_set.insert(elem);
                } else {
                    chunked_set.insert(elem);
                }
            }
            3..6 => {
                // Remove from one of the sets.
                if domain_size == 0 {
                    continue;
                }
                let elem = rng.next() % domain_size;
                // Choose set to remove into.
                if rng.next_bool() {
                    dense_set.remove(elem);
                } else {
                    chunked_set.remove(elem);
                }
            }
            6 => {
                // Clear
                if rng.next_bool() {
                    dense_set.clear();
                } else {
                    chunked_set.clear();
                }
            }
            7 => {
                // Fill.
                if rng.next_bool() {
                    dense_set.insert_all(domain_size);
                } else {
                    chunked_set.insert_all();
                }
            }
            8 => {
                // Union
                let old_dense_set = dense_set.clone();
                let changed = dense_set.union(&chunked_set);
                assert_eq!(old_dense_set != dense_set, changed);
                assert!(dense_set.superset(&old_dense_set));
                assert!(chunked_set.iter().all(|x| dense_set.contains(x)));

                // Check that all the added elements come from `chunked_set`.
                let mut difference = dense_set.clone();
                difference.subtract(&old_dense_set);
                assert!(difference.iter().all(|x| chunked_set.contains(x)));
            }
            9 => {
                // Intersection
                let old_dense_set = dense_set.clone();
                let changed = dense_set.intersect(&chunked_set);
                assert_eq!(old_dense_set != dense_set, changed);
                assert!(old_dense_set.superset(&dense_set));
                assert!(dense_set.iter().all(|x| chunked_set.contains(x)));

                // Check that no of the removed elements comes from `chunked_set`.
                let mut difference = old_dense_set; // Just renaming.
                difference.subtract(&dense_set);
                assert!(difference.iter().all(|x| !chunked_set.contains(x)));
            }
            _ => unreachable!(),
        }
    }
}

#[test]
fn test_dense_bit_set() {
    assert_eq!(
        size_of::<DenseBitSet<usize>>(),
        size_of::<Word>(),
        "DenseBitSet should have the same size as a Word"
    );

    test_with_domain_size(0);
    test_with_domain_size(1);
    test_with_domain_size(63);
    test_with_domain_size(64);
    test_with_domain_size(65);
    test_with_domain_size(127);
    test_with_domain_size(128);
    test_with_domain_size(129);

    test_relations_with_chunked_set(0);
    test_relations_with_chunked_set(1);
    test_relations_with_chunked_set(CHUNK_BITS - 1);
    test_relations_with_chunked_set(CHUNK_BITS);
    test_relations_with_chunked_set(CHUNK_BITS + 2);
    test_relations_with_chunked_set(3 * CHUNK_BITS - 2);
    test_relations_with_chunked_set(3 * CHUNK_BITS);
    test_relations_with_chunked_set(3 * CHUNK_BITS + 1);
}

#[test]
fn test_growable_bit_set() {
    const TEST_ITERATIONS: u32 = 512;
    const MAX_ELEMS: usize = 314;

    let mut set = GrowableBitSet::<usize>::new_empty();
    let mut reference_set = BTreeSet::<usize>::new();

    let mut rng = Rng::new(42);

    for _ in 0..TEST_ITERATIONS {
        match rng.next() % 100 {
            0..30 => {
                // Insert an element in the `0..=(DenseBitSet::INLINE_CAPACITY + 2)` range.
                let elem = rng.next() % (DenseBitSet::<usize>::INLINE_CAPACITY + 3);
                set.insert(elem);
                reference_set.insert(elem);
            }
            30..50 => {
                // Insert an element in the `0..MAX_ELEMS` range.
                let elem = rng.next() % MAX_ELEMS;
                set.insert(elem);
                reference_set.insert(elem);
            }
            50..70 => {
                // Remove an existing element.
                let len = set.len();
                if len == 0 {
                    continue;
                }
                let elem = set.iter().nth(rng.next() % len).unwrap();
                set.remove(elem);
                reference_set.remove(&elem);
            }
            70..90 => {
                // Remove an arbitrary element in the `0..MAX_ELEMS` range.
                let elem = rng.next() % MAX_ELEMS;
                set.remove(elem);
                reference_set.remove(&elem);
            }
            90..100 => {
                // Make sure the `with_capacity()` function works.
                let capacity = rng.next() % MAX_ELEMS;
                set = GrowableBitSet::with_capacity(capacity);
                reference_set.clear();
            }
            _ => unreachable!(),
        }

        // Check the `is_empty()` function.
        assert_eq!(set.is_empty(), reference_set.is_empty());

        // Check the `iter` function.
        assert!(set.iter().eq(reference_set.iter().copied()));

        // Check the contains function with a 20 % probability.
        if rng.next() % 5 == 0 {
            for x in 0..MAX_ELEMS {
                assert_eq!(set.contains(x), reference_set.contains(&x));
            }
        }
    }
}

#[test]
fn test_new_filled() {
    for i in 0..128 {
        let idx_buf = DenseBitSet::new_filled(i);
        let elems: Vec<usize> = idx_buf.iter().collect();
        let expected: Vec<usize> = (0..i).collect();
        assert_eq!(elems, expected);
    }
}

#[test]
fn bitset_iter_works() {
    let mut bitset: DenseBitSet<usize> = DenseBitSet::new_empty(100);
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
    let mut bitset: DenseBitSet<usize> = DenseBitSet::new_empty(320);
    bitset.insert(0);
    bitset.insert(127);
    bitset.insert(191);
    bitset.insert(255);
    bitset.insert(319);
    assert_eq!(bitset.iter().collect::<Vec<_>>(), [0, 127, 191, 255, 319]);
}

#[test]
fn bitset_clone_from() {
    let mut a: DenseBitSet<usize> = DenseBitSet::new_empty(10);
    a.insert(4);
    a.insert(7);
    a.insert(9);

    let mut b = DenseBitSet::new_empty(2);
    b.clone_from(&a);
    assert!(b.capacity() >= 10);
    assert_eq!(b.iter().collect::<Vec<_>>(), [4, 7, 9]);

    b.clone_from(&DenseBitSet::new_empty(40));
    assert!(b.capacity() >= 40);
    assert_eq!(b.iter().collect::<Vec<_>>(), []);
}

#[test]
fn union_two_sets() {
    let mut set1: DenseBitSet<usize> = DenseBitSet::new_empty(65);
    let mut set2: DenseBitSet<usize> = DenseBitSet::new_empty(65);
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
fn union_not() {
    let mut a = DenseBitSet::<usize>::new_empty(100);
    let mut b = DenseBitSet::<usize>::new_empty(100);

    a.insert(3);
    a.insert(5);
    a.insert(80);
    a.insert(81);

    b.insert(5); // Already in `a`.
    b.insert(7);
    b.insert(63);
    b.insert(81); // Already in `a`.
    b.insert(90);

    a.union_not(&b, 100);

    // After union-not, `a` should contain all values in the domain, except for
    // the ones that are in `b` and were _not_ already in `a`.
    assert_eq!(
        a.iter().collect::<Vec<_>>(),
        (0usize..100).filter(|&x| !matches!(x, 7 | 63 | 90)).collect::<Vec<_>>(),
    );
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

fn with_elements_standard(elements: &[usize], domain_size: usize) -> DenseBitSet<usize> {
    let mut s = DenseBitSet::new_empty(domain_size);
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

    let mut disjoint: DenseBitSet<usize> = DenseBitSet::new_empty(100);
    disjoint.insert(33);

    let mut superset = DenseBitSet::new_empty(100);
    superset.insert(22);
    superset.insert(75);
    superset.insert(33);

    let mut subset = DenseBitSet::new_empty(100);
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
    fn check_range(domain: usize, range: Range<usize>) {
        let mut set = DenseBitSet::new_empty(domain);
        set.insert_range(range.clone());
        for i in set.iter() {
            assert!(range.contains(&i));
        }
        for i in range.clone() {
            assert!(set.contains(i), "{} in {:?}, inserted {:?}", i, set, range);
        }
    }

    #[track_caller]
    fn check_range_inclusive(domain: usize, range: RangeInclusive<usize>) {
        let mut set = DenseBitSet::new_empty(domain);
        set.insert_range_inclusive(range.clone());
        for i in set.iter() {
            assert!(range.contains(&i));
        }
        for i in range.clone() {
            assert!(set.contains(i), "{} in {:?}, inserted {:?}", i, set, range);
        }
    }

    check_range(300, 10..10);
    check_range(300, WORD_BITS..WORD_BITS * 2);
    check_range(300, WORD_BITS - 1..WORD_BITS * 2);
    check_range(300, WORD_BITS - 1..WORD_BITS);
    check_range(300, 10..100);
    check_range(300, 10..30);
    check_range(300, 0..5);
    check_range(300, 0..250);
    check_range(300, 200..250);

    check_range_inclusive(300, 10..=10);
    check_range_inclusive(300, WORD_BITS..=WORD_BITS * 2);
    check_range_inclusive(300, WORD_BITS - 1..=WORD_BITS * 2);
    check_range_inclusive(300, WORD_BITS - 1..=WORD_BITS);
    check_range_inclusive(300, 10..=100);
    check_range_inclusive(300, 10..=30);
    check_range_inclusive(300, 0..=5);
    check_range_inclusive(300, 0..=250);
    check_range_inclusive(300, 200..=250);

    for i in 0..WORD_BITS * 2 {
        for j in i..WORD_BITS * 2 {
            check_range(WORD_BITS * 2, i..j);
            check_range_inclusive(WORD_BITS * 2, i..=j);
            check_range(300, i..j);
            check_range_inclusive(300, i..=j);
        }
    }
}

#[test]
fn dense_last_set_before() {
    fn easy(set: &DenseBitSet<usize>, needle: impl RangeBounds<usize>) -> Option<usize> {
        let mut last_leq = None;
        for e in set.iter() {
            if needle.contains(&e) {
                last_leq = Some(e);
            }
        }
        last_leq
    }

    #[track_caller]
    fn cmp(set: &DenseBitSet<usize>, needle: RangeInclusive<usize>) {
        assert_eq!(
            set.last_set_in(needle.clone()),
            easy(set, needle.clone()),
            "{:?} in {:?}",
            needle,
            set
        );
    }
    let mut set = DenseBitSet::new_empty(300);
    cmp(&set, 50..=50);
    set.insert(WORD_BITS);
    cmp(&set, WORD_BITS..=WORD_BITS);
    set.insert(WORD_BITS - 1);
    cmp(&set, 0..=WORD_BITS - 1);
    cmp(&set, 0..=5);
    cmp(&set, 10..=99);
    set.insert(100);
    cmp(&set, 100..=119);
    cmp(&set, 99..=99);
    cmp(&set, 99..=100);

    for i in 0..=WORD_BITS * 2 {
        for j in i..=WORD_BITS * 2 {
            for k in 0..WORD_BITS * 2 {
                let mut set = DenseBitSet::new_empty(300);
                cmp(&set, i..=j);
                set.insert(k);
                cmp(&set, i..=j);
            }
        }
    }
}

#[bench]
fn bench_insert(b: &mut Bencher) {
    let mut bs = DenseBitSet::new_filled(99999usize);
    b.iter(|| {
        black_box(bs.insert(black_box(100u32)));
    });
}

#[bench]
fn bench_remove(b: &mut Bencher) {
    let mut bs = DenseBitSet::new_filled(99999usize);
    b.iter(|| {
        black_box(bs.remove(black_box(100u32)));
    });
}

#[bench]
fn bench_iter(b: &mut Bencher) {
    let bs = DenseBitSet::new_filled(99999usize);
    b.iter(|| {
        bs.iter().map(|b: usize| black_box(b)).for_each(drop);
    });
}

#[bench]
fn bench_intersect(b: &mut Bencher) {
    let mut ba: DenseBitSet<u32> = DenseBitSet::new_filled(99999usize);
    let bb = DenseBitSet::new_filled(99999usize);
    b.iter(|| {
        ba.intersect(black_box(&bb));
    });
}
