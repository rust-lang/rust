use core::iter::TrustedLen;

use super::*;
use crate::testing::macros::struct_with_counted_drop;

#[bench]
fn bench_push_back_100(b: &mut test::Bencher) {
    let mut deq = VecDeque::with_capacity(101);
    b.iter(|| {
        for i in 0..100 {
            deq.push_back(i);
        }
        deq.head = 0;
        deq.len = 0;
    })
}

#[bench]
fn bench_push_front_100(b: &mut test::Bencher) {
    let mut deq = VecDeque::with_capacity(101);
    b.iter(|| {
        for i in 0..100 {
            deq.push_front(i);
        }
        deq.head = 0;
        deq.len = 0;
    })
}

#[bench]
fn bench_pop_back_100(b: &mut test::Bencher) {
    let size = 100;
    let mut deq = VecDeque::<i32>::with_capacity(size + 1);
    // We'll mess with private state to pretend like `deq` is filled.
    // Make sure the buffer is initialized so that we don't read uninit memory.
    unsafe { deq.ptr().write_bytes(0u8, size + 1) };

    b.iter(|| {
        deq.head = 0;
        deq.len = 100;
        while !deq.is_empty() {
            test::black_box(deq.pop_back());
        }
    })
}

#[bench]
fn bench_retain_whole_10000(b: &mut test::Bencher) {
    let size = if cfg!(miri) { 1000 } else { 100000 };
    let v = (1..size).collect::<VecDeque<u32>>();

    b.iter(|| {
        let mut v = v.clone();
        v.retain(|x| *x > 0)
    })
}

#[bench]
fn bench_retain_odd_10000(b: &mut test::Bencher) {
    let size = if cfg!(miri) { 1000 } else { 100000 };
    let v = (1..size).collect::<VecDeque<u32>>();

    b.iter(|| {
        let mut v = v.clone();
        v.retain(|x| x & 1 == 0)
    })
}

#[bench]
fn bench_retain_half_10000(b: &mut test::Bencher) {
    let size = if cfg!(miri) { 1000 } else { 100000 };
    let v = (1..size).collect::<VecDeque<u32>>();

    b.iter(|| {
        let mut v = v.clone();
        v.retain(|x| *x > size / 2)
    })
}

#[bench]
fn bench_pop_front_100(b: &mut test::Bencher) {
    let size = 100;
    let mut deq = VecDeque::<i32>::with_capacity(size + 1);
    // We'll mess with private state to pretend like `deq` is filled.
    // Make sure the buffer is initialized so that we don't read uninit memory.
    unsafe { deq.ptr().write_bytes(0u8, size + 1) };

    b.iter(|| {
        deq.head = 0;
        deq.len = 100;
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
            for head_pos in 0..usable_cap {
                tester.head = head_pos;
                tester.len = 0;
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
                assert!(tester.head <= tester.capacity());
                assert!(tester.len <= tester.capacity());
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
        for head_pos in 0..cap {
            for to_insert in 0..len {
                tester.head = head_pos;
                tester.len = 0;
                for i in 0..len {
                    if i != to_insert {
                        tester.push_back(i);
                    }
                }
                tester.insert(to_insert, to_insert);
                assert!(tester.head <= tester.capacity());
                assert!(tester.len <= tester.capacity());
                assert_eq!(tester, expected);
            }
        }
    }
}

#[test]
fn test_get() {
    let mut tester = VecDeque::new();
    tester.push_back(1);
    tester.push_back(2);
    tester.push_back(3);

    assert_eq!(tester.len(), 3);

    assert_eq!(tester.get(1), Some(&2));
    assert_eq!(tester.get(2), Some(&3));
    assert_eq!(tester.get(0), Some(&1));
    assert_eq!(tester.get(3), None);

    tester.remove(0);

    assert_eq!(tester.len(), 2);
    assert_eq!(tester.get(0), Some(&2));
    assert_eq!(tester.get(1), Some(&3));
    assert_eq!(tester.get(2), None);
}

#[test]
fn test_get_mut() {
    let mut tester = VecDeque::new();
    tester.push_back(1);
    tester.push_back(2);
    tester.push_back(3);

    assert_eq!(tester.len(), 3);

    if let Some(elem) = tester.get_mut(0) {
        assert_eq!(*elem, 1);
        *elem = 10;
    }

    if let Some(elem) = tester.get_mut(2) {
        assert_eq!(*elem, 3);
        *elem = 30;
    }

    assert_eq!(tester.get(0), Some(&10));
    assert_eq!(tester.get(2), Some(&30));
    assert_eq!(tester.get_mut(3), None);

    tester.remove(2);

    assert_eq!(tester.len(), 2);
    assert_eq!(tester.get(0), Some(&10));
    assert_eq!(tester.get(1), Some(&2));
    assert_eq!(tester.get(2), None);
}

#[test]
fn test_swap() {
    let mut tester = VecDeque::new();
    tester.push_back(1);
    tester.push_back(2);
    tester.push_back(3);

    assert_eq!(tester, [1, 2, 3]);

    tester.swap(0, 0);
    assert_eq!(tester, [1, 2, 3]);
    tester.swap(0, 1);
    assert_eq!(tester, [2, 1, 3]);
    tester.swap(2, 1);
    assert_eq!(tester, [2, 3, 1]);
    tester.swap(1, 2);
    assert_eq!(tester, [2, 1, 3]);
    tester.swap(0, 2);
    assert_eq!(tester, [3, 1, 2]);
    tester.swap(2, 2);
    assert_eq!(tester, [3, 1, 2]);
}

#[test]
#[should_panic = "assertion failed: j < self.len()"]
fn test_swap_panic() {
    let mut tester = VecDeque::new();
    tester.push_back(1);
    tester.push_back(2);
    tester.push_back(3);
    tester.swap(2, 3);
}

#[test]
fn test_reserve_exact() {
    let mut tester: VecDeque<i32> = VecDeque::with_capacity(1);
    assert_eq!(tester.capacity(), 1);
    tester.reserve_exact(50);
    assert_eq!(tester.capacity(), 50);
    tester.reserve_exact(40);
    // reserving won't shrink the buffer
    assert_eq!(tester.capacity(), 50);
    tester.reserve_exact(200);
    assert_eq!(tester.capacity(), 200);
}

#[test]
#[should_panic = "capacity overflow"]
fn test_reserve_exact_panic() {
    let mut tester: VecDeque<i32> = VecDeque::new();
    tester.reserve_exact(usize::MAX);
}

#[test]
fn test_try_reserve_exact() {
    let mut tester: VecDeque<i32> = VecDeque::with_capacity(1);
    assert!(tester.capacity() == 1);
    assert_eq!(tester.try_reserve_exact(100), Ok(()));
    assert!(tester.capacity() >= 100);
    assert_eq!(tester.try_reserve_exact(50), Ok(()));
    assert!(tester.capacity() >= 100);
    assert_eq!(tester.try_reserve_exact(200), Ok(()));
    assert!(tester.capacity() >= 200);
    assert_eq!(tester.try_reserve_exact(0), Ok(()));
    assert!(tester.capacity() >= 200);
    assert!(tester.try_reserve_exact(usize::MAX).is_err());
}

#[test]
fn test_try_reserve() {
    let mut tester: VecDeque<i32> = VecDeque::with_capacity(1);
    assert!(tester.capacity() == 1);
    assert_eq!(tester.try_reserve(100), Ok(()));
    assert!(tester.capacity() >= 100);
    assert_eq!(tester.try_reserve(50), Ok(()));
    assert!(tester.capacity() >= 100);
    assert_eq!(tester.try_reserve(200), Ok(()));
    assert!(tester.capacity() >= 200);
    assert_eq!(tester.try_reserve(0), Ok(()));
    assert!(tester.capacity() >= 200);
    assert!(tester.try_reserve(usize::MAX).is_err());
}

#[test]
fn test_contains() {
    let mut tester = VecDeque::new();
    tester.push_back(1);
    tester.push_back(2);
    tester.push_back(3);

    assert!(tester.contains(&1));
    assert!(tester.contains(&3));
    assert!(!tester.contains(&0));
    assert!(!tester.contains(&4));
    tester.remove(0);
    assert!(!tester.contains(&1));
    assert!(tester.contains(&2));
    assert!(tester.contains(&3));
}

#[test]
fn test_rotate_left_right() {
    let mut tester: VecDeque<_> = (1..=10).collect();
    tester.reserve(1);

    assert_eq!(tester.len(), 10);

    tester.rotate_left(0);
    assert_eq!(tester, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);

    tester.rotate_right(0);
    assert_eq!(tester, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);

    tester.rotate_left(3);
    assert_eq!(tester, [4, 5, 6, 7, 8, 9, 10, 1, 2, 3]);

    tester.rotate_right(5);
    assert_eq!(tester, [9, 10, 1, 2, 3, 4, 5, 6, 7, 8]);

    tester.rotate_left(tester.len());
    assert_eq!(tester, [9, 10, 1, 2, 3, 4, 5, 6, 7, 8]);

    tester.rotate_right(tester.len());
    assert_eq!(tester, [9, 10, 1, 2, 3, 4, 5, 6, 7, 8]);

    tester.rotate_left(1);
    assert_eq!(tester, [10, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
}

#[test]
#[should_panic = "assertion failed: n <= self.len()"]
fn test_rotate_left_panic() {
    let mut tester: VecDeque<_> = (1..=10).collect();
    tester.rotate_left(tester.len() + 1);
}

#[test]
#[should_panic = "assertion failed: n <= self.len()"]
fn test_rotate_right_panic() {
    let mut tester: VecDeque<_> = (1..=10).collect();
    tester.rotate_right(tester.len() + 1);
}

#[test]
fn test_binary_search() {
    // If the givin VecDeque is not sorted, the returned result is unspecified and meaningless,
    // as this method performs a binary search.

    let tester: VecDeque<_> = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55].into();

    assert_eq!(tester.binary_search(&0), Ok(0));
    assert_eq!(tester.binary_search(&5), Ok(5));
    assert_eq!(tester.binary_search(&55), Ok(10));
    assert_eq!(tester.binary_search(&4), Err(5));
    assert_eq!(tester.binary_search(&-1), Err(0));
    assert!(matches!(tester.binary_search(&1), Ok(1..=2)));

    let tester: VecDeque<_> = [1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3].into();
    assert_eq!(tester.binary_search(&1), Ok(0));
    assert!(matches!(tester.binary_search(&2), Ok(1..=4)));
    assert!(matches!(tester.binary_search(&3), Ok(5..=13)));
    assert_eq!(tester.binary_search(&-2), Err(0));
    assert_eq!(tester.binary_search(&0), Err(0));
    assert_eq!(tester.binary_search(&4), Err(14));
    assert_eq!(tester.binary_search(&5), Err(14));
}

#[test]
fn test_binary_search_by() {
    // If the givin VecDeque is not sorted, the returned result is unspecified and meaningless,
    // as this method performs a binary search.

    let tester: VecDeque<_> = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55].into();

    assert_eq!(tester.binary_search_by(|x| x.cmp(&0)), Ok(0));
    assert_eq!(tester.binary_search_by(|x| x.cmp(&5)), Ok(5));
    assert_eq!(tester.binary_search_by(|x| x.cmp(&55)), Ok(10));
    assert_eq!(tester.binary_search_by(|x| x.cmp(&4)), Err(5));
    assert_eq!(tester.binary_search_by(|x| x.cmp(&-1)), Err(0));
    assert!(matches!(tester.binary_search_by(|x| x.cmp(&1)), Ok(1..=2)));
}

#[test]
fn test_binary_search_key() {
    // If the givin VecDeque is not sorted, the returned result is unspecified and meaningless,
    // as this method performs a binary search.

    let tester: VecDeque<_> = [
        (-1, 0),
        (2, 10),
        (6, 5),
        (7, 1),
        (8, 10),
        (10, 2),
        (20, 3),
        (24, 5),
        (25, 18),
        (28, 13),
        (31, 21),
        (32, 4),
        (54, 25),
    ]
    .into();

    assert_eq!(tester.binary_search_by_key(&-1, |&(a, _b)| a), Ok(0));
    assert_eq!(tester.binary_search_by_key(&8, |&(a, _b)| a), Ok(4));
    assert_eq!(tester.binary_search_by_key(&25, |&(a, _b)| a), Ok(8));
    assert_eq!(tester.binary_search_by_key(&54, |&(a, _b)| a), Ok(12));
    assert_eq!(tester.binary_search_by_key(&-2, |&(a, _b)| a), Err(0));
    assert_eq!(tester.binary_search_by_key(&1, |&(a, _b)| a), Err(1));
    assert_eq!(tester.binary_search_by_key(&4, |&(a, _b)| a), Err(2));
    assert_eq!(tester.binary_search_by_key(&13, |&(a, _b)| a), Err(6));
    assert_eq!(tester.binary_search_by_key(&55, |&(a, _b)| a), Err(13));
    assert_eq!(tester.binary_search_by_key(&100, |&(a, _b)| a), Err(13));

    let tester: VecDeque<_> = [
        (0, 0),
        (2, 1),
        (6, 1),
        (5, 1),
        (3, 1),
        (1, 2),
        (2, 3),
        (4, 5),
        (5, 8),
        (8, 13),
        (1, 21),
        (2, 34),
        (4, 55),
    ]
    .into();

    assert_eq!(tester.binary_search_by_key(&0, |&(_a, b)| b), Ok(0));
    assert!(matches!(tester.binary_search_by_key(&1, |&(_a, b)| b), Ok(1..=4)));
    assert_eq!(tester.binary_search_by_key(&8, |&(_a, b)| b), Ok(8));
    assert_eq!(tester.binary_search_by_key(&13, |&(_a, b)| b), Ok(9));
    assert_eq!(tester.binary_search_by_key(&55, |&(_a, b)| b), Ok(12));
    assert_eq!(tester.binary_search_by_key(&-1, |&(_a, b)| b), Err(0));
    assert_eq!(tester.binary_search_by_key(&4, |&(_a, b)| b), Err(7));
    assert_eq!(tester.binary_search_by_key(&56, |&(_a, b)| b), Err(13));
    assert_eq!(tester.binary_search_by_key(&100, |&(_a, b)| b), Err(13));
}

#[test]
fn make_contiguous_big_head() {
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

    let expected_start = tester.as_slices().1.len();
    tester.make_contiguous();
    assert_eq!(tester.head, expected_start);
    assert_eq!((&[9, 8, 7, 6, 5, 4, 3, 0, 1, 2] as &[_], &[] as &[_]), tester.as_slices());
}

#[test]
fn make_contiguous_big_tail() {
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
    assert_eq!(tester.head, expected_start);
    assert_eq!((&[9, 8, 0, 1, 2, 3, 4, 5, 6, 7] as &[_], &[] as &[_]), tester.as_slices());
}

#[test]
fn make_contiguous_small_free() {
    let mut tester = VecDeque::with_capacity(16);

    for i in b'A'..b'I' {
        tester.push_back(i as char);
    }

    for i in b'I'..b'N' {
        tester.push_front(i as char);
    }

    assert_eq!(tester, ['M', 'L', 'K', 'J', 'I', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']);

    // ABCDEFGH...MLKJI
    let expected_start = 0;
    tester.make_contiguous();
    assert_eq!(tester.head, expected_start);
    assert_eq!(
        (&['M', 'L', 'K', 'J', 'I', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'] as &[_], &[] as &[_]),
        tester.as_slices()
    );

    tester.clear();
    for i in b'I'..b'N' {
        tester.push_back(i as char);
    }

    for i in b'A'..b'I' {
        tester.push_front(i as char);
    }

    // IJKLM...HGFEDCBA
    let expected_start = 3;
    tester.make_contiguous();
    assert_eq!(tester.head, expected_start);
    assert_eq!(
        (&['H', 'G', 'F', 'E', 'D', 'C', 'B', 'A', 'I', 'J', 'K', 'L', 'M'] as &[_], &[] as &[_]),
        tester.as_slices()
    );
}

#[test]
fn make_contiguous_head_to_end() {
    let mut tester = VecDeque::with_capacity(16);

    for i in b'A'..b'L' {
        tester.push_back(i as char);
    }

    for i in b'L'..b'Q' {
        tester.push_front(i as char);
    }

    assert_eq!(
        tester,
        ['P', 'O', 'N', 'M', 'L', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K']
    );

    // ABCDEFGHIJKPONML
    let expected_start = 0;
    tester.make_contiguous();
    assert_eq!(tester.head, expected_start);
    assert_eq!(
        (
            &['P', 'O', 'N', 'M', 'L', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K']
                as &[_],
            &[] as &[_]
        ),
        tester.as_slices()
    );

    tester.clear();
    for i in b'L'..b'Q' {
        tester.push_back(i as char);
    }

    for i in b'A'..b'L' {
        tester.push_front(i as char);
    }

    // LMNOPKJIHGFEDCBA
    let expected_start = 0;
    tester.make_contiguous();
    assert_eq!(tester.head, expected_start);
    assert_eq!(
        (
            &['K', 'J', 'I', 'H', 'G', 'F', 'E', 'D', 'C', 'B', 'A', 'L', 'M', 'N', 'O', 'P']
                as &[_],
            &[] as &[_]
        ),
        tester.as_slices()
    );
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
        for head_pos in 0..cap {
            for to_remove in 0..=len {
                tester.head = head_pos;
                tester.len = 0;
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
                assert!(tester.head <= tester.capacity());
                assert!(tester.len <= tester.capacity());
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
        for head in 0..=cap {
            for start in 0..=len {
                for end in start..=len {
                    tester.head = head;
                    tester.len = 0;
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
        for head in 0..=cap {
            for start in 0..=len {
                for end in start..=len {
                    tester.head = head;
                    tester.len = 0;
                    for i in 0..len {
                        tester.push_back(i);
                    }

                    let head_was = tester.head;
                    let len_was = tester.len;

                    // Check that we iterate over the correct values
                    let range: VecDeque<_> = tester.range_mut(start..end).map(|v| *v).collect();
                    let expected: VecDeque<_> = (start..end).collect();
                    assert_eq!(range, expected);

                    // We shouldn't have changed the capacity or made the
                    // head or tail out of bounds
                    assert_eq!(tester.capacity(), cap);
                    assert_eq!(tester.head, head_was);
                    assert_eq!(tester.len, len_was);
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
        for head in 0..cap {
            for drain_start in 0..=len {
                for drain_end in drain_start..=len {
                    tester.head = head;
                    tester.len = 0;
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
                    assert!(tester.head <= tester.capacity());
                    assert!(tester.len <= tester.capacity());

                    // We should see the correct values in the VecDeque
                    let expected: VecDeque<_> = (0..drain_start).chain(drain_end..len).collect();
                    assert_eq!(expected, tester);
                }
            }
        }
    }
}

#[test]
fn issue_108453() {
    let mut deque = VecDeque::with_capacity(10);

    deque.push_back(1u8);
    deque.push_back(2);
    deque.push_back(3);

    deque.push_front(10);
    deque.push_front(9);

    deque.shrink_to(9);

    assert_eq!(deque.into_iter().collect::<Vec<_>>(), vec![9, 10, 1, 2, 3]);
}

#[test]
fn test_shrink_to() {
    // test deques with capacity 16 with all possible head positions, lengths and target capacities.
    let cap = 16;

    for len in 0..cap {
        for head in 0..cap {
            let expected = (1..=len).collect::<VecDeque<_>>();

            for target_cap in len..cap {
                let mut deque = VecDeque::with_capacity(cap);
                // currently, `with_capacity` always allocates the exact capacity if it's greater than 8.
                assert_eq!(deque.capacity(), cap);

                // we can let the head point anywhere in the buffer since the deque is empty.
                deque.head = head;
                deque.extend(1..=len);

                deque.shrink_to(target_cap);

                assert_eq!(deque, expected);
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
        for head_pos in 0..=max_cap {
            tester.reserve(head_pos);
            tester.head = head_pos;
            tester.len = 0;
            tester.reserve(63);
            for i in 0..len {
                tester.push_back(i);
            }
            tester.shrink_to_fit();
            assert!(tester.capacity() <= cap);
            assert!(tester.head <= tester.capacity());
            assert!(tester.len <= tester.capacity());
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

            for head_pos in 0..cap {
                tester.head = head_pos;
                tester.len = 0;
                for i in 0..len {
                    tester.push_back(i);
                }
                let result = tester.split_off(at);
                assert!(tester.head <= tester.capacity());
                assert!(tester.len <= tester.capacity());
                assert!(result.head <= result.capacity());
                assert!(result.len <= result.capacity());
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
            assert_eq!(vd.len(), vec.len());
            assert!(vd.into_iter().eq(vec));
        }
    }
}

#[test]
fn test_extend_basic() {
    test_extend_impl(false);
}

#[test]
fn test_extend_trusted_len() {
    test_extend_impl(true);
}

fn test_extend_impl(trusted_len: bool) {
    struct VecDequeTester {
        test: VecDeque<usize>,
        expected: VecDeque<usize>,
        trusted_len: bool,
    }

    impl VecDequeTester {
        fn new(trusted_len: bool) -> Self {
            Self { test: VecDeque::new(), expected: VecDeque::new(), trusted_len }
        }

        fn test_extend<I>(&mut self, iter: I)
        where
            I: Iterator<Item = usize> + TrustedLen + Clone,
        {
            struct BasicIterator<I>(I);
            impl<I> Iterator for BasicIterator<I>
            where
                I: Iterator<Item = usize>,
            {
                type Item = usize;

                fn next(&mut self) -> Option<Self::Item> {
                    self.0.next()
                }
            }

            if self.trusted_len {
                self.test.extend(iter.clone());
            } else {
                self.test.extend(BasicIterator(iter.clone()));
            }

            for item in iter {
                self.expected.push_back(item)
            }

            assert_eq!(self.test, self.expected);
        }

        fn drain<R: RangeBounds<usize> + Clone>(&mut self, range: R) {
            self.test.drain(range.clone());
            self.expected.drain(range);

            assert_eq!(self.test, self.expected);
        }

        fn clear(&mut self) {
            self.test.clear();
            self.expected.clear();
        }

        fn remaining_capacity(&self) -> usize {
            self.test.capacity() - self.test.len()
        }
    }

    let mut tester = VecDequeTester::new(trusted_len);

    // Initial capacity
    tester.test_extend(0..tester.remaining_capacity());

    // Grow
    tester.test_extend(1024..2048);

    // Wrap around
    tester.drain(..128);

    tester.test_extend(0..tester.remaining_capacity());

    // Continue
    tester.drain(256..);
    tester.test_extend(4096..8196);

    tester.clear();

    // Start again
    tester.test_extend(0..32);
}

#[test]
fn test_from_array() {
    fn test<const N: usize>() {
        let mut array: [usize; N] = [0; N];

        for i in 0..N {
            array[i] = i;
        }

        let deq: VecDeque<_> = array.into();

        for i in 0..N {
            assert_eq!(deq[i], i);
        }

        assert_eq!(deq.len(), N);
    }
    test::<0>();
    test::<1>();
    test::<2>();
    test::<32>();
    test::<35>();
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
    struct_with_counted_drop!(Elem, DROPS);

    const LEN: usize = 5;
    for push_front in 0..=LEN {
        let mut tester = VecDeque::with_capacity(LEN);
        for index in 0..LEN {
            if index < push_front {
                tester.push_front(Elem);
            } else {
                tester.push_back(Elem);
            }
        }
        assert_eq!(DROPS.get(), 0);
        tester.truncate(3);
        assert_eq!(DROPS.get(), 2);
        tester.truncate(0);
        assert_eq!(DROPS.get(), 5);
        DROPS.set(0);
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
