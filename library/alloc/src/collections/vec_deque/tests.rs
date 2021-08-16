use super::*;

#[bench]
#[cfg_attr(miri, ignore)] // isolated Miri does not support benchmarks
fn bench_push_back_100(b: &mut test::Bencher) {
    let mut deq = VecDeque::with_capacity(101);
    b.iter(|| {
        for i in 0..100 {
            deq.push_back(i);
        }
        deq.head = 0;
        deq.tail = 0;
    })
}

#[bench]
#[cfg_attr(miri, ignore)] // isolated Miri does not support benchmarks
fn bench_push_front_100(b: &mut test::Bencher) {
    let mut deq = VecDeque::with_capacity(101);
    b.iter(|| {
        for i in 0..100 {
            deq.push_front(i);
        }
        deq.head = 0;
        deq.tail = 0;
    })
}

#[bench]
#[cfg_attr(miri, ignore)] // isolated Miri does not support benchmarks
fn bench_pop_back_100(b: &mut test::Bencher) {
    let mut deq = VecDeque::<i32>::with_capacity(101);

    b.iter(|| {
        deq.head = 100;
        deq.tail = 0;
        while !deq.is_empty() {
            test::black_box(deq.pop_back());
        }
    })
}

#[bench]
#[cfg_attr(miri, ignore)] // isolated Miri does not support benchmarks
fn bench_retain_whole_10000(b: &mut test::Bencher) {
    let v = (1..100000).collect::<VecDeque<u32>>();

    b.iter(|| {
        let mut v = v.clone();
        v.retain(|x| *x > 0)
    })
}

#[bench]
#[cfg_attr(miri, ignore)] // isolated Miri does not support benchmarks
fn bench_retain_odd_10000(b: &mut test::Bencher) {
    let v = (1..100000).collect::<VecDeque<u32>>();

    b.iter(|| {
        let mut v = v.clone();
        v.retain(|x| x & 1 == 0)
    })
}

#[bench]
#[cfg_attr(miri, ignore)] // isolated Miri does not support benchmarks
fn bench_retain_half_10000(b: &mut test::Bencher) {
    let v = (1..100000).collect::<VecDeque<u32>>();

    b.iter(|| {
        let mut v = v.clone();
        v.retain(|x| *x > 50000)
    })
}

#[bench]
#[cfg_attr(miri, ignore)] // isolated Miri does not support benchmarks
fn bench_pop_front_100(b: &mut test::Bencher) {
    let mut deq = VecDeque::<i32>::with_capacity(101);

    b.iter(|| {
        deq.head = 100;
        deq.tail = 0;
        while !deq.is_empty() {
            test::black_box(deq.pop_front());
        }
    })
}



#[test]
fn test_swap_front_back_remove() {
    fn test(back: bool) {
        // This test checks that every single combination of tail position and length is tested.
        // Capacity 15 should be large enough to cover every case.
        let mut tester = VecDeque::with_capacity(15);
        let usable_cap = tester.capacity();
        let final_len = usable_cap / 2;

        for len in 0..final_len {
            let expected: VecDeque<_> =
                if back { (0..len).collect() } else { (0..len).rev().collect() };
            for tail_pos in 0..usable_cap {
                tester.tail = tail_pos;
                tester.head = tail_pos;
                if back {
                    for i in 0..len * 2 {
                        tester.push_front(i);
                    }
                    for i in 0..len {
                        assert_eq!(tester.swap_remove_back(i), Some(len * 2 - 1 - i));
                    }
                } else {
                    for i in 0..len * 2 {
                        tester.push_back(i);
                    }
                    for i in 0..len {
                        let idx = tester.len() - 1 - i;
                        assert_eq!(tester.swap_remove_front(idx), Some(len * 2 - 1 - i));
                    }
                }
                assert!(tester.tail < tester.cap());
                assert!(tester.head < tester.cap());
                assert_eq!(tester, expected);
            }
        }
    }
    test(true);
    test(false);
}

#[test]
fn test_insert() {
    // This test checks that every single combination of tail position, length, and
    // insertion position is tested. Capacity 15 should be large enough to cover every case.

    let mut tester = VecDeque::with_capacity(15);
    // can't guarantee we got 15, so have to get what we got.
    // 15 would be great, but we will definitely get 2^k - 1, for k >= 4, or else
    // this test isn't covering what it wants to
    let cap = tester.capacity();

    // len is the length *after* insertion
    let minlen = if cfg!(miri) { cap - 1 } else { 1 }; // Miri is too slow
    for len in minlen..cap {
        // 0, 1, 2, .., len - 1
        let expected = (0..).take(len).collect::<VecDeque<_>>();
        for tail_pos in 0..cap {
            for to_insert in 0..len {
                tester.tail = tail_pos;
                tester.head = tail_pos;
                for i in 0..len {
                    if i != to_insert {
                        tester.push_back(i);
                    }
                }
                tester.insert(to_insert, to_insert);
                assert!(tester.tail < tester.cap());
                assert!(tester.head < tester.cap());
                assert_eq!(tester, expected);
            }
        }
    }
}

#[test]
fn make_contiguous_big_tail() {
    let mut tester = VecDeque::with_capacity(15);

    for i in 0..3 {
        tester.push_back(i);
    }

    for i in 3..10 {
        tester.push_front(i);
    }

    // 012......9876543
    assert_eq!(tester.capacity(), 15);
    assert_eq!((&[9, 8, 7, 6, 5, 4, 3] as &[_], &[0, 1, 2] as &[_]), tester.as_slices());

    let expected_start = tester.head;
    tester.make_contiguous();
    assert_eq!(tester.tail, expected_start);
    assert_eq!((&[9, 8, 7, 6, 5, 4, 3, 0, 1, 2] as &[_], &[] as &[_]), tester.as_slices());
}

#[test]
fn make_contiguous_big_head() {
    let mut tester = VecDeque::with_capacity(15);

    for i in 0..8 {
        tester.push_back(i);
    }

    for i in 8..10 {
        tester.push_front(i);
    }

    // 01234567......98
    let expected_start = 0;
    tester.make_contiguous();
    assert_eq!(tester.tail, expected_start);
    assert_eq!((&[9, 8, 0, 1, 2, 3, 4, 5, 6, 7] as &[_], &[] as &[_]), tester.as_slices());
}

#[test]
fn make_contiguous_small_free() {
    let mut tester = VecDeque::with_capacity(15);

    for i in 'A' as u8..'I' as u8 {
        tester.push_back(i as char);
    }

    for i in 'I' as u8..'N' as u8 {
        tester.push_front(i as char);
    }

    // ABCDEFGH...MLKJI
    let expected_start = 0;
    tester.make_contiguous();
    assert_eq!(tester.tail, expected_start);
    assert_eq!(
        (&['M', 'L', 'K', 'J', 'I', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'] as &[_], &[] as &[_]),
        tester.as_slices()
    );

    tester.clear();
    for i in 'I' as u8..'N' as u8 {
        tester.push_back(i as char);
    }

    for i in 'A' as u8..'I' as u8 {
        tester.push_front(i as char);
    }

    // IJKLM...HGFEDCBA
    let expected_start = 0;
    tester.make_contiguous();
    assert_eq!(tester.tail, expected_start);
    assert_eq!(
        (&['H', 'G', 'F', 'E', 'D', 'C', 'B', 'A', 'I', 'J', 'K', 'L', 'M'] as &[_], &[] as &[_]),
        tester.as_slices()
    );
}

#[test]
fn make_contiguous_head_to_end() {
    let mut dq = VecDeque::with_capacity(3);
    dq.push_front('B');
    dq.push_front('A');
    dq.push_back('C');
    dq.make_contiguous();
    let expected_tail = 0;
    let expected_head = 3;
    assert_eq!(expected_tail, dq.tail);
    assert_eq!(expected_head, dq.head);
    assert_eq!((&['A', 'B', 'C'] as &[_], &[] as &[_]), dq.as_slices());
}

#[test]
fn make_contiguous_head_to_end_2() {
    // Another test case for #79808, taken from #80293.

    let mut dq = VecDeque::from_iter(0..6);
    dq.pop_front();
    dq.pop_front();
    dq.push_back(6);
    dq.push_back(7);
    dq.push_back(8);
    dq.make_contiguous();
    let collected: Vec<_> = dq.iter().copied().collect();
    assert_eq!(dq.as_slices(), (&collected[..], &[] as &[_]));
}

#[test]
fn test_remove() {
    // This test checks that every single combination of tail position, length, and
    // removal position is tested. Capacity 15 should be large enough to cover every case.

    let mut tester = VecDeque::with_capacity(15);
    // can't guarantee we got 15, so have to get what we got.
    // 15 would be great, but we will definitely get 2^k - 1, for k >= 4, or else
    // this test isn't covering what it wants to
    let cap = tester.capacity();

    // len is the length *after* removal
    let minlen = if cfg!(miri) { cap - 2 } else { 0 }; // Miri is too slow
    for len in minlen..cap - 1 {
        // 0, 1, 2, .., len - 1
        let expected = (0..).take(len).collect::<VecDeque<_>>();
        for tail_pos in 0..cap {
            for to_remove in 0..=len {
                tester.tail = tail_pos;
                tester.head = tail_pos;
                for i in 0..len {
                    if i == to_remove {
                        tester.push_back(1234);
                    }
                    tester.push_back(i);
                }
                if to_remove == len {
                    tester.push_back(1234);
                }
                tester.remove(to_remove);
                assert!(tester.tail < tester.cap());
                assert!(tester.head < tester.cap());
                assert_eq!(tester, expected);
            }
        }
    }
}

#[test]
fn test_range() {
    let mut tester: VecDeque<usize> = VecDeque::with_capacity(7);

    let cap = tester.capacity();
    let minlen = if cfg!(miri) { cap - 1 } else { 0 }; // Miri is too slow
    for len in minlen..=cap {
        for tail in 0..=cap {
            for start in 0..=len {
                for end in start..=len {
                    tester.tail = tail;
                    tester.head = tail;
                    for i in 0..len {
                        tester.push_back(i);
                    }

                    // Check that we iterate over the correct values
                    let range: VecDeque<_> = tester.range(start..end).copied().collect();
                    let expected: VecDeque<_> = (start..end).collect();
                    assert_eq!(range, expected);
                }
            }
        }
    }
}

#[test]
fn test_range_mut() {
    let mut tester: VecDeque<usize> = VecDeque::with_capacity(7);

    let cap = tester.capacity();
    for len in 0..=cap {
        for tail in 0..=cap {
            for start in 0..=len {
                for end in start..=len {
                    tester.tail = tail;
                    tester.head = tail;
                    for i in 0..len {
                        tester.push_back(i);
                    }

                    let head_was = tester.head;
                    let tail_was = tester.tail;

                    // Check that we iterate over the correct values
                    let range: VecDeque<_> = tester.range_mut(start..end).map(|v| *v).collect();
                    let expected: VecDeque<_> = (start..end).collect();
                    assert_eq!(range, expected);

                    // We shouldn't have changed the capacity or made the
                    // head or tail out of bounds
                    assert_eq!(tester.capacity(), cap);
                    assert_eq!(tester.tail, tail_was);
                    assert_eq!(tester.head, head_was);
                }
            }
        }
    }
}

#[test]
fn test_drain() {
    let mut tester: VecDeque<usize> = VecDeque::with_capacity(7);

    let cap = tester.capacity();
    for len in 0..=cap {
        for tail in 0..=cap {
            for drain_start in 0..=len {
                for drain_end in drain_start..=len {
                    tester.tail = tail;
                    tester.head = tail;
                    for i in 0..len {
                        tester.push_back(i);
                    }

                    // Check that we drain the correct values
                    let drained: VecDeque<_> = tester.drain(drain_start..drain_end).collect();
                    let drained_expected: VecDeque<_> = (drain_start..drain_end).collect();
                    assert_eq!(drained, drained_expected);

                    // We shouldn't have changed the capacity or made the
                    // head or tail out of bounds
                    assert_eq!(tester.capacity(), cap);
                    assert!(tester.tail < tester.cap());
                    assert!(tester.head < tester.cap());

                    // We should see the correct values in the VecDeque
                    let expected: VecDeque<_> = (0..drain_start).chain(drain_end..len).collect();
                    assert_eq!(expected, tester);
                }
            }
        }
    }
}

#[test]
fn test_shrink_to_fit() {
    // This test checks that every single combination of head and tail position,
    // is tested. Capacity 15 should be large enough to cover every case.

    let mut tester = VecDeque::with_capacity(15);
    // can't guarantee we got 15, so have to get what we got.
    // 15 would be great, but we will definitely get 2^k - 1, for k >= 4, or else
    // this test isn't covering what it wants to
    let cap = tester.capacity();
    tester.reserve(63);
    let max_cap = tester.capacity();

    for len in 0..=cap {
        // 0, 1, 2, .., len - 1
        let expected = (0..).take(len).collect::<VecDeque<_>>();
        for tail_pos in 0..=max_cap {
            tester.tail = tail_pos;
            tester.head = tail_pos;
            tester.reserve(63);
            for i in 0..len {
                tester.push_back(i);
            }
            tester.shrink_to_fit();
            assert!(tester.capacity() <= cap);
            assert!(tester.tail < tester.cap());
            assert!(tester.head < tester.cap());
            assert_eq!(tester, expected);
        }
    }
}

#[test]
fn test_split_off() {
    // This test checks that every single combination of tail position, length, and
    // split position is tested. Capacity 15 should be large enough to cover every case.

    let mut tester = VecDeque::with_capacity(15);
    // can't guarantee we got 15, so have to get what we got.
    // 15 would be great, but we will definitely get 2^k - 1, for k >= 4, or else
    // this test isn't covering what it wants to
    let cap = tester.capacity();

    // len is the length *before* splitting
    let minlen = if cfg!(miri) { cap - 1 } else { 0 }; // Miri is too slow
    for len in minlen..cap {
        // index to split at
        for at in 0..=len {
            // 0, 1, 2, .., at - 1 (may be empty)
            let expected_self = (0..).take(at).collect::<VecDeque<_>>();
            // at, at + 1, .., len - 1 (may be empty)
            let expected_other = (at..).take(len - at).collect::<VecDeque<_>>();

            for tail_pos in 0..cap {
                tester.tail = tail_pos;
                tester.head = tail_pos;
                for i in 0..len {
                    tester.push_back(i);
                }
                let result = tester.split_off(at);
                assert!(tester.tail < tester.cap());
                assert!(tester.head < tester.cap());
                assert!(result.tail < result.cap());
                assert!(result.head < result.cap());
                assert_eq!(tester, expected_self);
                assert_eq!(result, expected_other);
            }
        }
    }
}

#[test]
fn test_from_vec() {
    use crate::vec::Vec;
    for cap in 0..35 {
        for len in 0..=cap {
            let mut vec = Vec::with_capacity(cap);
            vec.extend(0..len);

            let vd = VecDeque::from(vec.clone());
            assert!(vd.cap().is_power_of_two());
            assert_eq!(vd.len(), vec.len());
            assert!(vd.into_iter().eq(vec));
        }
    }

    let vec = Vec::from([(); MAXIMUM_ZST_CAPACITY - 1]);
    let vd = VecDeque::from(vec.clone());
    assert!(vd.cap().is_power_of_two());
    assert_eq!(vd.len(), vec.len());
}

#[test]
#[should_panic = "capacity overflow"]
fn test_from_vec_zst_overflow() {
    use crate::vec::Vec;
    let vec = Vec::from([(); MAXIMUM_ZST_CAPACITY]);
    let vd = VecDeque::from(vec.clone()); // no room for +1
    assert!(vd.cap().is_power_of_two());
    assert_eq!(vd.len(), vec.len());
}

#[test]
fn test_vec_from_vecdeque() {
    use crate::vec::Vec;

    fn create_vec_and_test_convert(capacity: usize, offset: usize, len: usize) {
        let mut vd = VecDeque::with_capacity(capacity);
        for _ in 0..offset {
            vd.push_back(0);
            vd.pop_front();
        }
        vd.extend(0..len);

        let vec: Vec<_> = Vec::from(vd.clone());
        assert_eq!(vec.len(), vd.len());
        assert!(vec.into_iter().eq(vd));
    }

    // Miri is too slow
    let max_pwr = if cfg!(miri) { 5 } else { 7 };

    for cap_pwr in 0..max_pwr {
        // Make capacity as a (2^x)-1, so that the ring size is 2^x
        let cap = (2i32.pow(cap_pwr) - 1) as usize;

        // In these cases there is enough free space to solve it with copies
        for len in 0..((cap + 1) / 2) {
            // Test contiguous cases
            for offset in 0..(cap - len) {
                create_vec_and_test_convert(cap, offset, len)
            }

            // Test cases where block at end of buffer is bigger than block at start
            for offset in (cap - len)..(cap - (len / 2)) {
                create_vec_and_test_convert(cap, offset, len)
            }

            // Test cases where block at start of buffer is bigger than block at end
            for offset in (cap - (len / 2))..cap {
                create_vec_and_test_convert(cap, offset, len)
            }
        }

        // Now there's not (necessarily) space to straighten the ring with simple copies,
        // the ring will use swapping when:
        // (cap + 1 - offset) > (cap + 1 - len) && (len - (cap + 1 - offset)) > (cap + 1 - len))
        //  right block size  >   free space    &&      left block size       >    free space
        for len in ((cap + 1) / 2)..cap {
            // Test contiguous cases
            for offset in 0..(cap - len) {
                create_vec_and_test_convert(cap, offset, len)
            }

            // Test cases where block at end of buffer is bigger than block at start
            for offset in (cap - len)..(cap - (len / 2)) {
                create_vec_and_test_convert(cap, offset, len)
            }

            // Test cases where block at start of buffer is bigger than block at end
            for offset in (cap - (len / 2))..cap {
                create_vec_and_test_convert(cap, offset, len)
            }
        }
    }
}

#[test]
fn test_clone_from() {
    let m = vec![1; 8];
    let n = vec![2; 12];
    let limit = if cfg!(miri) { 4 } else { 8 }; // Miri is too slow
    for pfv in 0..limit {
        for pfu in 0..limit {
            for longer in 0..2 {
                let (vr, ur) = if longer == 0 { (&m, &n) } else { (&n, &m) };
                let mut v = VecDeque::from(vr.clone());
                for _ in 0..pfv {
                    v.push_front(1);
                }
                let mut u = VecDeque::from(ur.clone());
                for _ in 0..pfu {
                    u.push_front(2);
                }
                v.clone_from(&u);
                assert_eq!(&v, &u);
            }
        }
    }
}

#[test]
fn test_vec_deque_truncate_drop() {
    static mut DROPS: u32 = 0;
    #[derive(Clone)]
    struct Elem(i32);
    impl Drop for Elem {
        fn drop(&mut self) {
            unsafe {
                DROPS += 1;
            }
        }
    }

    let v = vec![Elem(1), Elem(2), Elem(3), Elem(4), Elem(5)];
    for push_front in 0..=v.len() {
        let v = v.clone();
        let mut tester = VecDeque::with_capacity(5);
        for (index, elem) in v.into_iter().enumerate() {
            if index < push_front {
                tester.push_front(elem);
            } else {
                tester.push_back(elem);
            }
        }
        assert_eq!(unsafe { DROPS }, 0);
        tester.truncate(3);
        assert_eq!(unsafe { DROPS }, 2);
        tester.truncate(0);
        assert_eq!(unsafe { DROPS }, 5);
        unsafe {
            DROPS = 0;
        }
    }
}

#[test]
fn issue_53529() {
    use crate::boxed::Box;

    let mut dst = VecDeque::new();
    dst.push_front(Box::new(1));
    dst.push_front(Box::new(2));
    assert_eq!(*dst.pop_back().unwrap(), 1);

    let mut src = VecDeque::new();
    src.push_front(Box::new(2));
    dst.append(&mut src);
    for a in dst {
        assert_eq!(*a, 2);
    }
}

#[test]
fn issue_80303() {
    use core::iter;
    use core::num::Wrapping;

    // This is a valid, albeit rather bad hash function implementation.
    struct SimpleHasher(Wrapping<u64>);

    impl Hasher for SimpleHasher {
        fn finish(&self) -> u64 {
            self.0.0
        }

        fn write(&mut self, bytes: &[u8]) {
            // This particular implementation hashes value 24 in addition to bytes.
            // Such an implementation is valid as Hasher only guarantees equivalence
            // for the exact same set of calls to its methods.
            for &v in iter::once(&24).chain(bytes) {
                self.0 = Wrapping(31) * self.0 + Wrapping(u64::from(v));
            }
        }
    }

    fn hash_code(value: impl Hash) -> u64 {
        let mut hasher = SimpleHasher(Wrapping(1));
        value.hash(&mut hasher);
        hasher.finish()
    }

    // This creates two deques for which values returned by as_slices
    // method differ.
    let vda: VecDeque<u8> = (0..10).collect();
    let mut vdb = VecDeque::with_capacity(10);
    vdb.extend(5..10);
    (0..5).rev().for_each(|elem| vdb.push_front(elem));
    assert_ne!(vda.as_slices(), vdb.as_slices());
    assert_eq!(vda, vdb);
    assert_eq!(hash_code(vda), hash_code(vdb));
}
