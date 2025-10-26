use core::cell::Cell;
use core::cmp::Ordering;
use core::mem::MaybeUninit;
use core::num::NonZero;
use core::ops::{Range, RangeInclusive};
use core::slice;

#[test]
fn test_position() {
    let b = [1, 2, 3, 5, 5];
    assert_eq!(b.iter().position(|&v| v == 9), None);
    assert_eq!(b.iter().position(|&v| v == 5), Some(3));
    assert_eq!(b.iter().position(|&v| v == 3), Some(2));
    assert_eq!(b.iter().position(|&v| v == 0), None);
}

#[test]
fn test_rposition() {
    let b = [1, 2, 3, 5, 5];
    assert_eq!(b.iter().rposition(|&v| v == 9), None);
    assert_eq!(b.iter().rposition(|&v| v == 5), Some(4));
    assert_eq!(b.iter().rposition(|&v| v == 3), Some(2));
    assert_eq!(b.iter().rposition(|&v| v == 0), None);
}

#[test]
fn test_binary_search() {
    let b: [i32; 0] = [];
    assert_eq!(b.binary_search(&5), Err(0));

    let b = [4];
    assert_eq!(b.binary_search(&3), Err(0));
    assert_eq!(b.binary_search(&4), Ok(0));
    assert_eq!(b.binary_search(&5), Err(1));

    let b = [1, 2, 4, 6, 8, 9];
    assert_eq!(b.binary_search(&5), Err(3));
    assert_eq!(b.binary_search(&6), Ok(3));
    assert_eq!(b.binary_search(&7), Err(4));
    assert_eq!(b.binary_search(&8), Ok(4));

    let b = [1, 2, 4, 5, 6, 8];
    assert_eq!(b.binary_search(&9), Err(6));

    let b = [1, 2, 4, 6, 7, 8, 9];
    assert_eq!(b.binary_search(&6), Ok(3));
    assert_eq!(b.binary_search(&5), Err(3));
    assert_eq!(b.binary_search(&8), Ok(5));

    let b = [1, 2, 4, 5, 6, 8, 9];
    assert_eq!(b.binary_search(&7), Err(5));
    assert_eq!(b.binary_search(&0), Err(0));

    let b = [1, 3, 3, 3, 7];
    assert_eq!(b.binary_search(&0), Err(0));
    assert_eq!(b.binary_search(&1), Ok(0));
    assert_eq!(b.binary_search(&2), Err(1));
    assert!(match b.binary_search(&3) {
        Ok(1..=3) => true,
        _ => false,
    });
    assert!(match b.binary_search(&3) {
        Ok(1..=3) => true,
        _ => false,
    });
    assert_eq!(b.binary_search(&4), Err(4));
    assert_eq!(b.binary_search(&5), Err(4));
    assert_eq!(b.binary_search(&6), Err(4));
    assert_eq!(b.binary_search(&7), Ok(4));
    assert_eq!(b.binary_search(&8), Err(5));

    let b = [(); usize::MAX];
    assert_eq!(b.binary_search(&()), Ok(usize::MAX - 1));
}

#[test]
fn test_binary_search_by_overflow() {
    let b = [(); usize::MAX];
    assert_eq!(b.binary_search_by(|_| Ordering::Equal), Ok(usize::MAX - 1));
    assert_eq!(b.binary_search_by(|_| Ordering::Greater), Err(0));
    assert_eq!(b.binary_search_by(|_| Ordering::Less), Err(usize::MAX));
}

#[test]
// Test implementation specific behavior when finding equivalent elements.
// It is ok to break this test but when you do a crater run is highly advisable.
fn test_binary_search_implementation_details() {
    let b = [1, 1, 2, 2, 3, 3, 3];
    assert_eq!(b.binary_search(&1), Ok(1));
    assert_eq!(b.binary_search(&2), Ok(3));
    assert_eq!(b.binary_search(&3), Ok(6));
    let b = [1, 1, 1, 1, 1, 3, 3, 3, 3];
    assert_eq!(b.binary_search(&1), Ok(4));
    assert_eq!(b.binary_search(&3), Ok(8));
    let b = [1, 1, 1, 1, 3, 3, 3, 3, 3];
    assert_eq!(b.binary_search(&1), Ok(3));
    assert_eq!(b.binary_search(&3), Ok(8));
}

#[test]
fn test_partition_point() {
    let b: [i32; 0] = [];
    assert_eq!(b.partition_point(|&x| x < 5), 0);

    let b = [4];
    assert_eq!(b.partition_point(|&x| x < 3), 0);
    assert_eq!(b.partition_point(|&x| x < 4), 0);
    assert_eq!(b.partition_point(|&x| x < 5), 1);

    let b = [1, 2, 4, 6, 8, 9];
    assert_eq!(b.partition_point(|&x| x < 5), 3);
    assert_eq!(b.partition_point(|&x| x < 6), 3);
    assert_eq!(b.partition_point(|&x| x < 7), 4);
    assert_eq!(b.partition_point(|&x| x < 8), 4);

    let b = [1, 2, 4, 5, 6, 8];
    assert_eq!(b.partition_point(|&x| x < 9), 6);

    let b = [1, 2, 4, 6, 7, 8, 9];
    assert_eq!(b.partition_point(|&x| x < 6), 3);
    assert_eq!(b.partition_point(|&x| x < 5), 3);
    assert_eq!(b.partition_point(|&x| x < 8), 5);

    let b = [1, 2, 4, 5, 6, 8, 9];
    assert_eq!(b.partition_point(|&x| x < 7), 5);
    assert_eq!(b.partition_point(|&x| x < 0), 0);

    let b = [1, 3, 3, 3, 7];
    assert_eq!(b.partition_point(|&x| x < 0), 0);
    assert_eq!(b.partition_point(|&x| x < 1), 0);
    assert_eq!(b.partition_point(|&x| x < 2), 1);
    assert_eq!(b.partition_point(|&x| x < 3), 1);
    assert_eq!(b.partition_point(|&x| x < 4), 4);
    assert_eq!(b.partition_point(|&x| x < 5), 4);
    assert_eq!(b.partition_point(|&x| x < 6), 4);
    assert_eq!(b.partition_point(|&x| x < 7), 4);
    assert_eq!(b.partition_point(|&x| x < 8), 5);
}

#[test]
fn test_iterator_advance_by() {
    let v = &[0, 1, 2, 3, 4];

    for i in 0..=v.len() {
        let mut iter = v.iter();
        assert_eq!(iter.advance_by(i), Ok(()));
        assert_eq!(iter.as_slice(), &v[i..]);
    }

    let mut iter = v.iter();
    assert_eq!(iter.advance_by(v.len() + 1), Err(NonZero::new(1).unwrap()));
    assert_eq!(iter.as_slice(), &[]);

    let mut iter = v.iter();
    assert_eq!(iter.advance_by(3), Ok(()));
    assert_eq!(iter.as_slice(), &v[3..]);
    assert_eq!(iter.advance_by(2), Ok(()));
    assert_eq!(iter.as_slice(), &[]);
    assert_eq!(iter.advance_by(0), Ok(()));
}

#[test]
fn test_iterator_advance_back_by() {
    let v = &[0, 1, 2, 3, 4];

    for i in 0..=v.len() {
        let mut iter = v.iter();
        assert_eq!(iter.advance_back_by(i), Ok(()));
        assert_eq!(iter.as_slice(), &v[..v.len() - i]);
    }

    let mut iter = v.iter();
    assert_eq!(iter.advance_back_by(v.len() + 1), Err(NonZero::new(1).unwrap()));
    assert_eq!(iter.as_slice(), &[]);

    let mut iter = v.iter();
    assert_eq!(iter.advance_back_by(3), Ok(()));
    assert_eq!(iter.as_slice(), &v[..v.len() - 3]);
    assert_eq!(iter.advance_back_by(2), Ok(()));
    assert_eq!(iter.as_slice(), &[]);
    assert_eq!(iter.advance_back_by(0), Ok(()));
}

#[test]
fn test_iterator_nth() {
    let v: &[_] = &[0, 1, 2, 3, 4];
    for i in 0..v.len() {
        assert_eq!(v.iter().nth(i).unwrap(), &v[i]);
    }
    assert_eq!(v.iter().nth(v.len()), None);

    let mut iter = v.iter();
    assert_eq!(iter.nth(2).unwrap(), &v[2]);
    assert_eq!(iter.nth(1).unwrap(), &v[4]);
}

#[test]
fn test_iterator_nth_back() {
    let v: &[_] = &[0, 1, 2, 3, 4];
    for i in 0..v.len() {
        assert_eq!(v.iter().nth_back(i).unwrap(), &v[v.len() - i - 1]);
    }
    assert_eq!(v.iter().nth_back(v.len()), None);

    let mut iter = v.iter();
    assert_eq!(iter.nth_back(2).unwrap(), &v[2]);
    assert_eq!(iter.nth_back(1).unwrap(), &v[0]);
}

#[test]
fn test_iterator_last() {
    let v: &[_] = &[0, 1, 2, 3, 4];
    assert_eq!(v.iter().last().unwrap(), &4);
    assert_eq!(v[..1].iter().last().unwrap(), &0);
}

#[test]
fn test_iterator_count() {
    let v: &[_] = &[0, 1, 2, 3, 4];
    assert_eq!(v.iter().count(), 5);

    let mut iter2 = v.iter();
    iter2.next();
    iter2.next();
    assert_eq!(iter2.count(), 3);
}

#[test]
fn test_chunks_count() {
    let v: &[i32] = &[0, 1, 2, 3, 4, 5];
    let c = v.chunks(3);
    assert_eq!(c.count(), 2);

    let v2: &[i32] = &[0, 1, 2, 3, 4];
    let c2 = v2.chunks(2);
    assert_eq!(c2.count(), 3);

    let v3: &[i32] = &[];
    let c3 = v3.chunks(2);
    assert_eq!(c3.count(), 0);
}

#[test]
fn test_chunks_nth() {
    let v: &[i32] = &[0, 1, 2, 3, 4, 5];
    let mut c = v.chunks(2);
    assert_eq!(c.nth(1).unwrap(), &[2, 3]);
    assert_eq!(c.next().unwrap(), &[4, 5]);

    let v2: &[i32] = &[0, 1, 2, 3, 4];
    let mut c2 = v2.chunks(3);
    assert_eq!(c2.nth(1).unwrap(), &[3, 4]);
    assert_eq!(c2.next(), None);
}

#[test]
fn test_chunks_next() {
    let v = [0, 1, 2, 3, 4, 5];
    let mut c = v.chunks(2);
    assert_eq!(c.next().unwrap(), &[0, 1]);
    assert_eq!(c.next().unwrap(), &[2, 3]);
    assert_eq!(c.next().unwrap(), &[4, 5]);
    assert_eq!(c.next(), None);

    let v = [0, 1, 2, 3, 4, 5, 6, 7];
    let mut c = v.chunks(3);
    assert_eq!(c.next().unwrap(), &[0, 1, 2]);
    assert_eq!(c.next().unwrap(), &[3, 4, 5]);
    assert_eq!(c.next().unwrap(), &[6, 7]);
    assert_eq!(c.next(), None);
}

#[test]
fn test_chunks_next_back() {
    let v = [0, 1, 2, 3, 4, 5];
    let mut c = v.chunks(2);
    assert_eq!(c.next_back().unwrap(), &[4, 5]);
    assert_eq!(c.next_back().unwrap(), &[2, 3]);
    assert_eq!(c.next_back().unwrap(), &[0, 1]);
    assert_eq!(c.next_back(), None);

    let v = [0, 1, 2, 3, 4, 5, 6, 7];
    let mut c = v.chunks(3);
    assert_eq!(c.next_back().unwrap(), &[6, 7]);
    assert_eq!(c.next_back().unwrap(), &[3, 4, 5]);
    assert_eq!(c.next_back().unwrap(), &[0, 1, 2]);
    assert_eq!(c.next_back(), None);
}

#[test]
fn test_chunks_nth_back() {
    let v: &[i32] = &[0, 1, 2, 3, 4, 5];
    let mut c = v.chunks(2);
    assert_eq!(c.nth_back(1).unwrap(), &[2, 3]);
    assert_eq!(c.next().unwrap(), &[0, 1]);
    assert_eq!(c.next(), None);

    let v2: &[i32] = &[0, 1, 2, 3, 4];
    let mut c2 = v2.chunks(3);
    assert_eq!(c2.nth_back(1).unwrap(), &[0, 1, 2]);
    assert_eq!(c2.next(), None);
    assert_eq!(c2.next_back(), None);

    let v3: &[i32] = &[0, 1, 2, 3, 4];
    let mut c3 = v3.chunks(10);
    assert_eq!(c3.nth_back(0).unwrap(), &[0, 1, 2, 3, 4]);
    assert_eq!(c3.next(), None);

    let v4: &[i32] = &[0, 1, 2];
    let mut c4 = v4.chunks(10);
    assert_eq!(c4.nth_back(1_000_000_000usize), None);
}

#[test]
fn test_chunks_last() {
    let v: &[i32] = &[0, 1, 2, 3, 4, 5];
    let c = v.chunks(2);
    assert_eq!(c.last().unwrap()[1], 5);

    let v2: &[i32] = &[0, 1, 2, 3, 4];
    let c2 = v2.chunks(2);
    assert_eq!(c2.last().unwrap()[0], 4);
}

#[test]
fn test_chunks_zip() {
    let v1: &[i32] = &[0, 1, 2, 3, 4];
    let v2: &[i32] = &[6, 7, 8, 9, 10];

    let res = v1
        .chunks(2)
        .zip(v2.chunks(2))
        .map(|(a, b)| a.iter().sum::<i32>() + b.iter().sum::<i32>())
        .collect::<Vec<_>>();
    assert_eq!(res, vec![14, 22, 14]);
}

#[test]
fn test_chunks_mut_count() {
    let v: &mut [i32] = &mut [0, 1, 2, 3, 4, 5];
    let c = v.chunks_mut(3);
    assert_eq!(c.count(), 2);

    let v2: &mut [i32] = &mut [0, 1, 2, 3, 4];
    let c2 = v2.chunks_mut(2);
    assert_eq!(c2.count(), 3);

    let v3: &mut [i32] = &mut [];
    let c3 = v3.chunks_mut(2);
    assert_eq!(c3.count(), 0);
}

#[test]
fn test_chunks_mut_nth() {
    let v: &mut [i32] = &mut [0, 1, 2, 3, 4, 5];
    let mut c = v.chunks_mut(2);
    assert_eq!(c.nth(1).unwrap(), &[2, 3]);
    assert_eq!(c.next().unwrap(), &[4, 5]);

    let v2: &mut [i32] = &mut [0, 1, 2, 3, 4];
    let mut c2 = v2.chunks_mut(3);
    assert_eq!(c2.nth(1).unwrap(), &[3, 4]);
    assert_eq!(c2.next(), None);
}

#[test]
fn test_chunks_mut_nth_back() {
    let v: &mut [i32] = &mut [0, 1, 2, 3, 4, 5];
    let mut c = v.chunks_mut(2);
    assert_eq!(c.nth_back(1).unwrap(), &[2, 3]);
    assert_eq!(c.next().unwrap(), &[0, 1]);

    let v1: &mut [i32] = &mut [0, 1, 2, 3, 4];
    let mut c1 = v1.chunks_mut(3);
    assert_eq!(c1.nth_back(1).unwrap(), &[0, 1, 2]);
    assert_eq!(c1.next(), None);

    let v3: &mut [i32] = &mut [0, 1, 2, 3, 4];
    let mut c3 = v3.chunks_mut(10);
    assert_eq!(c3.nth_back(0).unwrap(), &[0, 1, 2, 3, 4]);
    assert_eq!(c3.next(), None);

    let v4: &mut [i32] = &mut [0, 1, 2];
    let mut c4 = v4.chunks_mut(10);
    assert_eq!(c4.nth_back(1_000_000_000usize), None);
}

#[test]
fn test_chunks_mut_last() {
    let v: &mut [i32] = &mut [0, 1, 2, 3, 4, 5];
    let c = v.chunks_mut(2);
    assert_eq!(c.last().unwrap(), &[4, 5]);

    let v2: &mut [i32] = &mut [0, 1, 2, 3, 4];
    let c2 = v2.chunks_mut(2);
    assert_eq!(c2.last().unwrap(), &[4]);
}

#[test]
fn test_chunks_mut_zip() {
    let v1: &mut [i32] = &mut [0, 1, 2, 3, 4];
    let v2: &[i32] = &[6, 7, 8, 9, 10];

    for (a, b) in v1.chunks_mut(2).zip(v2.chunks(2)) {
        let sum = b.iter().sum::<i32>();
        for v in a {
            *v += sum;
        }
    }
    assert_eq!(v1, [13, 14, 19, 20, 14]);
}

#[test]
fn test_chunks_mut_zip_aliasing() {
    let v1: &mut [i32] = &mut [0, 1, 2, 3, 4];
    let v2: &[i32] = &[6, 7, 8, 9, 10];

    let mut it = v1.chunks_mut(2).zip(v2.chunks(2));
    let first = it.next().unwrap();
    let _ = it.next().unwrap();
    assert_eq!(first, (&mut [0, 1][..], &[6, 7][..]));
}

#[test]
fn test_chunks_exact_mut_zip_aliasing() {
    let v1: &mut [i32] = &mut [0, 1, 2, 3, 4];
    let v2: &[i32] = &[6, 7, 8, 9, 10];

    let mut it = v1.chunks_exact_mut(2).zip(v2.chunks(2));
    let first = it.next().unwrap();
    let _ = it.next().unwrap();
    assert_eq!(first, (&mut [0, 1][..], &[6, 7][..]));
}

#[test]
fn test_rchunks_mut_zip_aliasing() {
    let v1: &mut [i32] = &mut [0, 1, 2, 3, 4];
    let v2: &[i32] = &[6, 7, 8, 9, 10];

    let mut it = v1.rchunks_mut(2).zip(v2.chunks(2));
    let first = it.next().unwrap();
    let _ = it.next().unwrap();
    assert_eq!(first, (&mut [3, 4][..], &[6, 7][..]));
}

#[test]
fn test_rchunks_exact_mut_zip_aliasing() {
    let v1: &mut [i32] = &mut [0, 1, 2, 3, 4];
    let v2: &[i32] = &[6, 7, 8, 9, 10];

    let mut it = v1.rchunks_exact_mut(2).zip(v2.chunks(2));
    let first = it.next().unwrap();
    let _ = it.next().unwrap();
    assert_eq!(first, (&mut [3, 4][..], &[6, 7][..]));
}

#[test]
fn test_chunks_exact_count() {
    let v: &[i32] = &[0, 1, 2, 3, 4, 5];
    let c = v.chunks_exact(3);
    assert_eq!(c.count(), 2);

    let v2: &[i32] = &[0, 1, 2, 3, 4];
    let c2 = v2.chunks_exact(2);
    assert_eq!(c2.count(), 2);

    let v3: &[i32] = &[];
    let c3 = v3.chunks_exact(2);
    assert_eq!(c3.count(), 0);
}

#[test]
fn test_chunks_exact_nth() {
    let v: &[i32] = &[0, 1, 2, 3, 4, 5];
    let mut c = v.chunks_exact(2);
    assert_eq!(c.nth(1).unwrap(), &[2, 3]);
    assert_eq!(c.next().unwrap(), &[4, 5]);

    let v2: &[i32] = &[0, 1, 2, 3, 4, 5, 6];
    let mut c2 = v2.chunks_exact(3);
    assert_eq!(c2.nth(1).unwrap(), &[3, 4, 5]);
    assert_eq!(c2.next(), None);
}

#[test]
fn test_chunks_exact_nth_back() {
    let v: &[i32] = &[0, 1, 2, 3, 4, 5];
    let mut c = v.chunks_exact(2);
    assert_eq!(c.nth_back(1).unwrap(), &[2, 3]);
    assert_eq!(c.next().unwrap(), &[0, 1]);
    assert_eq!(c.next(), None);

    let v2: &[i32] = &[0, 1, 2, 3, 4];
    let mut c2 = v2.chunks_exact(3);
    assert_eq!(c2.nth_back(0).unwrap(), &[0, 1, 2]);
    assert_eq!(c2.next(), None);
    assert_eq!(c2.next_back(), None);

    let v3: &[i32] = &[0, 1, 2, 3, 4];
    let mut c3 = v3.chunks_exact(10);
    assert_eq!(c3.nth_back(0), None);
}

#[test]
fn test_chunks_exact_last() {
    let v: &[i32] = &[0, 1, 2, 3, 4, 5];
    let c = v.chunks_exact(2);
    assert_eq!(c.last().unwrap(), &[4, 5]);

    let v2: &[i32] = &[0, 1, 2, 3, 4];
    let c2 = v2.chunks_exact(2);
    assert_eq!(c2.last().unwrap(), &[2, 3]);
}

#[test]
fn test_chunks_exact_remainder() {
    let v: &[i32] = &[0, 1, 2, 3, 4];
    let c = v.chunks_exact(2);
    assert_eq!(c.remainder(), &[4]);
}

#[test]
fn test_chunks_exact_zip() {
    let v1: &[i32] = &[0, 1, 2, 3, 4];
    let v2: &[i32] = &[6, 7, 8, 9, 10];

    let res = v1
        .chunks_exact(2)
        .zip(v2.chunks_exact(2))
        .map(|(a, b)| a.iter().sum::<i32>() + b.iter().sum::<i32>())
        .collect::<Vec<_>>();
    assert_eq!(res, vec![14, 22]);
}

#[test]
fn test_chunks_exact_mut_count() {
    let v: &mut [i32] = &mut [0, 1, 2, 3, 4, 5];
    let c = v.chunks_exact_mut(3);
    assert_eq!(c.count(), 2);

    let v2: &mut [i32] = &mut [0, 1, 2, 3, 4];
    let c2 = v2.chunks_exact_mut(2);
    assert_eq!(c2.count(), 2);

    let v3: &mut [i32] = &mut [];
    let c3 = v3.chunks_exact_mut(2);
    assert_eq!(c3.count(), 0);
}

#[test]
fn test_chunks_exact_mut_nth() {
    let v: &mut [i32] = &mut [0, 1, 2, 3, 4, 5];
    let mut c = v.chunks_exact_mut(2);
    assert_eq!(c.nth(1).unwrap(), &[2, 3]);
    assert_eq!(c.next().unwrap(), &[4, 5]);

    let v2: &mut [i32] = &mut [0, 1, 2, 3, 4, 5, 6];
    let mut c2 = v2.chunks_exact_mut(3);
    assert_eq!(c2.nth(1).unwrap(), &[3, 4, 5]);
    assert_eq!(c2.next(), None);
}

#[test]
fn test_chunks_exact_mut_nth_back() {
    let v: &mut [i32] = &mut [0, 1, 2, 3, 4, 5];
    let mut c = v.chunks_exact_mut(2);
    assert_eq!(c.nth_back(1).unwrap(), &[2, 3]);
    assert_eq!(c.next().unwrap(), &[0, 1]);
    assert_eq!(c.next(), None);

    let v2: &mut [i32] = &mut [0, 1, 2, 3, 4];
    let mut c2 = v2.chunks_exact_mut(3);
    assert_eq!(c2.nth_back(0).unwrap(), &[0, 1, 2]);
    assert_eq!(c2.next(), None);
    assert_eq!(c2.next_back(), None);

    let v3: &mut [i32] = &mut [0, 1, 2, 3, 4];
    let mut c3 = v3.chunks_exact_mut(10);
    assert_eq!(c3.nth_back(0), None);
}

#[test]
fn test_chunks_exact_mut_last() {
    let v: &mut [i32] = &mut [0, 1, 2, 3, 4, 5];
    let c = v.chunks_exact_mut(2);
    assert_eq!(c.last().unwrap(), &[4, 5]);

    let v2: &mut [i32] = &mut [0, 1, 2, 3, 4];
    let c2 = v2.chunks_exact_mut(2);
    assert_eq!(c2.last().unwrap(), &[2, 3]);
}

#[test]
fn test_chunks_exact_mut_remainder() {
    let v: &mut [i32] = &mut [0, 1, 2, 3, 4];
    let c = v.chunks_exact_mut(2);
    assert_eq!(c.into_remainder(), &[4]);
}

#[test]
fn test_chunks_exact_mut_zip() {
    let v1: &mut [i32] = &mut [0, 1, 2, 3, 4];
    let v2: &[i32] = &[6, 7, 8, 9, 10];

    for (a, b) in v1.chunks_exact_mut(2).zip(v2.chunks_exact(2)) {
        let sum = b.iter().sum::<i32>();
        for v in a {
            *v += sum;
        }
    }
    assert_eq!(v1, [13, 14, 19, 20, 4]);
}

#[test]
fn test_array_windows_infer() {
    let v: &[i32] = &[0, 1, 0, 1];
    assert_eq!(v.array_windows::<2>().count(), 3);
    let c = v.array_windows();
    for &[a, b] in c {
        assert_eq!(a + b, 1);
    }

    let v2: &[i32] = &[0, 1, 2, 3, 4, 5, 6];
    let total = v2.array_windows().map(|&[a, b, c]| a + b + c).sum::<i32>();
    assert_eq!(total, 3 + 6 + 9 + 12 + 15);
}

#[test]
fn test_array_windows_count() {
    let v: &[i32] = &[0, 1, 2, 3, 4, 5];
    let c = v.array_windows::<3>();
    assert_eq!(c.count(), 4);

    let v2: &[i32] = &[0, 1, 2, 3, 4];
    let c2 = v2.array_windows::<6>();
    assert_eq!(c2.count(), 0);

    let v3: &[i32] = &[];
    let c3 = v3.array_windows::<2>();
    assert_eq!(c3.count(), 0);

    let v4: &[()] = &[(); usize::MAX];
    let c4 = v4.array_windows::<1>();
    assert_eq!(c4.count(), usize::MAX);
}

#[test]
fn test_array_windows_nth() {
    let v: &[i32] = &[0, 1, 2, 3, 4, 5];
    let snd = v.array_windows::<4>().nth(1);
    assert_eq!(snd, Some(&[1, 2, 3, 4]));
    let mut arr_windows = v.array_windows::<2>();
    assert_ne!(arr_windows.nth(0), arr_windows.nth(0));
    let last = v.array_windows::<3>().last();
    assert_eq!(last, Some(&[3, 4, 5]));
}

#[test]
fn test_array_windows_nth_back() {
    let v: &[i32] = &[0, 1, 2, 3, 4, 5];
    let snd = v.array_windows::<4>().nth_back(1);
    assert_eq!(snd, Some(&[1, 2, 3, 4]));
    let mut arr_windows = v.array_windows::<2>();
    assert_ne!(arr_windows.nth_back(0), arr_windows.nth_back(0));
}

#[test]
fn test_rchunks_count() {
    let v: &[i32] = &[0, 1, 2, 3, 4, 5];
    let c = v.rchunks(3);
    assert_eq!(c.count(), 2);

    let v2: &[i32] = &[0, 1, 2, 3, 4];
    let c2 = v2.rchunks(2);
    assert_eq!(c2.count(), 3);

    let v3: &[i32] = &[];
    let c3 = v3.rchunks(2);
    assert_eq!(c3.count(), 0);
}

#[test]
fn test_rchunks_nth() {
    let v: &[i32] = &[0, 1, 2, 3, 4, 5];
    let mut c = v.rchunks(2);
    assert_eq!(c.nth(1).unwrap(), &[2, 3]);
    assert_eq!(c.next().unwrap(), &[0, 1]);

    let v2: &[i32] = &[0, 1, 2, 3, 4];
    let mut c2 = v2.rchunks(3);
    assert_eq!(c2.nth(1).unwrap(), &[0, 1]);
    assert_eq!(c2.next(), None);
}

#[test]
fn test_rchunks_nth_back() {
    let v: &[i32] = &[0, 1, 2, 3, 4, 5];
    let mut c = v.rchunks(2);
    assert_eq!(c.nth_back(1).unwrap(), &[2, 3]);
    assert_eq!(c.next_back().unwrap(), &[4, 5]);

    let v2: &[i32] = &[0, 1, 2, 3, 4];
    let mut c2 = v2.rchunks(3);
    assert_eq!(c2.nth_back(1).unwrap(), &[2, 3, 4]);
    assert_eq!(c2.next_back(), None);
}

#[test]
fn test_rchunks_next() {
    let v = [0, 1, 2, 3, 4, 5];
    let mut c = v.rchunks(2);
    assert_eq!(c.next().unwrap(), &[4, 5]);
    assert_eq!(c.next().unwrap(), &[2, 3]);
    assert_eq!(c.next().unwrap(), &[0, 1]);
    assert_eq!(c.next(), None);

    let v = [0, 1, 2, 3, 4, 5, 6, 7];
    let mut c = v.rchunks(3);
    assert_eq!(c.next().unwrap(), &[5, 6, 7]);
    assert_eq!(c.next().unwrap(), &[2, 3, 4]);
    assert_eq!(c.next().unwrap(), &[0, 1]);
    assert_eq!(c.next(), None);
}

#[test]
fn test_rchunks_next_back() {
    let v = [0, 1, 2, 3, 4, 5];
    let mut c = v.rchunks(2);
    assert_eq!(c.next_back().unwrap(), &[0, 1]);
    assert_eq!(c.next_back().unwrap(), &[2, 3]);
    assert_eq!(c.next_back().unwrap(), &[4, 5]);
    assert_eq!(c.next_back(), None);

    let v = [0, 1, 2, 3, 4, 5, 6, 7];
    let mut c = v.rchunks(3);
    assert_eq!(c.next_back().unwrap(), &[0, 1]);
    assert_eq!(c.next_back().unwrap(), &[2, 3, 4]);
    assert_eq!(c.next_back().unwrap(), &[5, 6, 7]);
    assert_eq!(c.next_back(), None);
}

#[test]
fn test_rchunks_last() {
    let v: &[i32] = &[0, 1, 2, 3, 4, 5];
    let c = v.rchunks(2);
    assert_eq!(c.last().unwrap()[1], 1);

    let v2: &[i32] = &[0, 1, 2, 3, 4];
    let c2 = v2.rchunks(2);
    assert_eq!(c2.last().unwrap()[0], 0);
}

#[test]
fn test_rchunks_zip() {
    let v1: &[i32] = &[0, 1, 2, 3, 4];
    let v2: &[i32] = &[6, 7, 8, 9, 10];

    let res = v1
        .rchunks(2)
        .zip(v2.rchunks(2))
        .map(|(a, b)| a.iter().sum::<i32>() + b.iter().sum::<i32>())
        .collect::<Vec<_>>();
    assert_eq!(res, vec![26, 18, 6]);
}

#[test]
fn test_rchunks_mut_count() {
    let v: &mut [i32] = &mut [0, 1, 2, 3, 4, 5];
    let c = v.rchunks_mut(3);
    assert_eq!(c.count(), 2);

    let v2: &mut [i32] = &mut [0, 1, 2, 3, 4];
    let c2 = v2.rchunks_mut(2);
    assert_eq!(c2.count(), 3);

    let v3: &mut [i32] = &mut [];
    let c3 = v3.rchunks_mut(2);
    assert_eq!(c3.count(), 0);
}

#[test]
fn test_rchunks_mut_nth() {
    let v: &mut [i32] = &mut [0, 1, 2, 3, 4, 5];
    let mut c = v.rchunks_mut(2);
    assert_eq!(c.nth(1).unwrap(), &[2, 3]);
    assert_eq!(c.next().unwrap(), &[0, 1]);

    let v2: &mut [i32] = &mut [0, 1, 2, 3, 4];
    let mut c2 = v2.rchunks_mut(3);
    assert_eq!(c2.nth(1).unwrap(), &[0, 1]);
    assert_eq!(c2.next(), None);
}

#[test]
fn test_rchunks_mut_nth_back() {
    let v: &mut [i32] = &mut [0, 1, 2, 3, 4, 5];
    let mut c = v.rchunks_mut(2);
    assert_eq!(c.nth_back(1).unwrap(), &[2, 3]);
    assert_eq!(c.next_back().unwrap(), &[4, 5]);

    let v2: &mut [i32] = &mut [0, 1, 2, 3, 4];
    let mut c2 = v2.rchunks_mut(3);
    assert_eq!(c2.nth_back(1).unwrap(), &[2, 3, 4]);
    assert_eq!(c2.next_back(), None);
}

#[test]
fn test_rchunks_mut_next() {
    let mut v = [0, 1, 2, 3, 4, 5];
    let mut c = v.rchunks_mut(2);
    assert_eq!(c.next().unwrap(), &mut [4, 5]);
    assert_eq!(c.next().unwrap(), &mut [2, 3]);
    assert_eq!(c.next().unwrap(), &mut [0, 1]);
    assert_eq!(c.next(), None);

    let mut v = [0, 1, 2, 3, 4, 5, 6, 7];
    let mut c = v.rchunks_mut(3);
    assert_eq!(c.next().unwrap(), &mut [5, 6, 7]);
    assert_eq!(c.next().unwrap(), &mut [2, 3, 4]);
    assert_eq!(c.next().unwrap(), &mut [0, 1]);
    assert_eq!(c.next(), None);
}

#[test]
fn test_rchunks_mut_next_back() {
    let mut v = [0, 1, 2, 3, 4, 5];
    let mut c = v.rchunks_mut(2);
    assert_eq!(c.next_back().unwrap(), &mut [0, 1]);
    assert_eq!(c.next_back().unwrap(), &mut [2, 3]);
    assert_eq!(c.next_back().unwrap(), &mut [4, 5]);
    assert_eq!(c.next_back(), None);

    let mut v = [0, 1, 2, 3, 4, 5, 6, 7];
    let mut c = v.rchunks_mut(3);
    assert_eq!(c.next_back().unwrap(), &mut [0, 1]);
    assert_eq!(c.next_back().unwrap(), &mut [2, 3, 4]);
    assert_eq!(c.next_back().unwrap(), &mut [5, 6, 7]);
    assert_eq!(c.next_back(), None);
}

#[test]
fn test_rchunks_mut_last() {
    let v: &mut [i32] = &mut [0, 1, 2, 3, 4, 5];
    let c = v.rchunks_mut(2);
    assert_eq!(c.last().unwrap(), &[0, 1]);

    let v2: &mut [i32] = &mut [0, 1, 2, 3, 4];
    let c2 = v2.rchunks_mut(2);
    assert_eq!(c2.last().unwrap(), &[0]);
}

#[test]
fn test_rchunks_mut_zip() {
    let v1: &mut [i32] = &mut [0, 1, 2, 3, 4];
    let v2: &[i32] = &[6, 7, 8, 9, 10];

    for (a, b) in v1.rchunks_mut(2).zip(v2.rchunks(2)) {
        let sum = b.iter().sum::<i32>();
        for v in a {
            *v += sum;
        }
    }
    assert_eq!(v1, [6, 16, 17, 22, 23]);
}

#[test]
fn test_rchunks_exact_count() {
    let v: &[i32] = &[0, 1, 2, 3, 4, 5];
    let c = v.rchunks_exact(3);
    assert_eq!(c.count(), 2);

    let v2: &[i32] = &[0, 1, 2, 3, 4];
    let c2 = v2.rchunks_exact(2);
    assert_eq!(c2.count(), 2);

    let v3: &[i32] = &[];
    let c3 = v3.rchunks_exact(2);
    assert_eq!(c3.count(), 0);
}

#[test]
fn test_rchunks_exact_nth() {
    let v: &[i32] = &[0, 1, 2, 3, 4, 5];
    let mut c = v.rchunks_exact(2);
    assert_eq!(c.nth(1).unwrap(), &[2, 3]);
    assert_eq!(c.next().unwrap(), &[0, 1]);

    let v2: &[i32] = &[0, 1, 2, 3, 4, 5, 6];
    let mut c2 = v2.rchunks_exact(3);
    assert_eq!(c2.nth(1).unwrap(), &[1, 2, 3]);
    assert_eq!(c2.next(), None);
}

#[test]
fn test_rchunks_exact_nth_back() {
    let v: &[i32] = &[0, 1, 2, 3, 4, 5];
    let mut c = v.rchunks_exact(2);
    assert_eq!(c.nth_back(1).unwrap(), &[2, 3]);
    assert_eq!(c.next_back().unwrap(), &[4, 5]);

    let v2: &[i32] = &[0, 1, 2, 3, 4, 5, 6];
    let mut c2 = v2.rchunks_exact(3);
    assert_eq!(c2.nth_back(1).unwrap(), &[4, 5, 6]);
    assert_eq!(c2.next(), None);
}

#[test]
fn test_rchunks_exact_last() {
    let v: &[i32] = &[0, 1, 2, 3, 4, 5];
    let c = v.rchunks_exact(2);
    assert_eq!(c.last().unwrap(), &[0, 1]);

    let v2: &[i32] = &[0, 1, 2, 3, 4];
    let c2 = v2.rchunks_exact(2);
    assert_eq!(c2.last().unwrap(), &[1, 2]);
}

#[test]
fn test_rchunks_exact_remainder() {
    let v: &[i32] = &[0, 1, 2, 3, 4];
    let c = v.rchunks_exact(2);
    assert_eq!(c.remainder(), &[0]);
}

#[test]
fn test_rchunks_exact_zip() {
    let v1: &[i32] = &[0, 1, 2, 3, 4];
    let v2: &[i32] = &[6, 7, 8, 9, 10];

    let res = v1
        .rchunks_exact(2)
        .zip(v2.rchunks_exact(2))
        .map(|(a, b)| a.iter().sum::<i32>() + b.iter().sum::<i32>())
        .collect::<Vec<_>>();
    assert_eq!(res, vec![26, 18]);
}

#[test]
fn test_rchunks_exact_mut_count() {
    let v: &mut [i32] = &mut [0, 1, 2, 3, 4, 5];
    let c = v.rchunks_exact_mut(3);
    assert_eq!(c.count(), 2);

    let v2: &mut [i32] = &mut [0, 1, 2, 3, 4];
    let c2 = v2.rchunks_exact_mut(2);
    assert_eq!(c2.count(), 2);

    let v3: &mut [i32] = &mut [];
    let c3 = v3.rchunks_exact_mut(2);
    assert_eq!(c3.count(), 0);
}

#[test]
fn test_rchunks_exact_mut_nth() {
    let v: &mut [i32] = &mut [0, 1, 2, 3, 4, 5];
    let mut c = v.rchunks_exact_mut(2);
    assert_eq!(c.nth(1).unwrap(), &[2, 3]);
    assert_eq!(c.next().unwrap(), &[0, 1]);

    let v2: &mut [i32] = &mut [0, 1, 2, 3, 4, 5, 6];
    let mut c2 = v2.rchunks_exact_mut(3);
    assert_eq!(c2.nth(1).unwrap(), &[1, 2, 3]);
    assert_eq!(c2.next(), None);
}

#[test]
fn test_rchunks_exact_mut_nth_back() {
    let v: &mut [i32] = &mut [0, 1, 2, 3, 4, 5];
    let mut c = v.rchunks_exact_mut(2);
    assert_eq!(c.nth_back(1).unwrap(), &[2, 3]);
    assert_eq!(c.next_back().unwrap(), &[4, 5]);

    let v2: &mut [i32] = &mut [0, 1, 2, 3, 4, 5, 6];
    let mut c2 = v2.rchunks_exact_mut(3);
    assert_eq!(c2.nth_back(1).unwrap(), &[4, 5, 6]);
    assert_eq!(c2.next(), None);
}

#[test]
fn test_rchunks_exact_mut_last() {
    let v: &mut [i32] = &mut [0, 1, 2, 3, 4, 5];
    let c = v.rchunks_exact_mut(2);
    assert_eq!(c.last().unwrap(), &[0, 1]);

    let v2: &mut [i32] = &mut [0, 1, 2, 3, 4];
    let c2 = v2.rchunks_exact_mut(2);
    assert_eq!(c2.last().unwrap(), &[1, 2]);
}

#[test]
fn test_rchunks_exact_mut_remainder() {
    let v: &mut [i32] = &mut [0, 1, 2, 3, 4];
    let c = v.rchunks_exact_mut(2);
    assert_eq!(c.into_remainder(), &[0]);
}

#[test]
fn test_rchunks_exact_mut_zip() {
    let v1: &mut [i32] = &mut [0, 1, 2, 3, 4];
    let v2: &[i32] = &[6, 7, 8, 9, 10];

    for (a, b) in v1.rchunks_exact_mut(2).zip(v2.rchunks_exact(2)) {
        let sum = b.iter().sum::<i32>();
        for v in a {
            *v += sum;
        }
    }
    assert_eq!(v1, [0, 16, 17, 22, 23]);
}

#[test]
fn chunks_mut_are_send_and_sync() {
    use std::cell::Cell;
    use std::slice::{ChunksExactMut, ChunksMut, RChunksExactMut, RChunksMut};
    use std::sync::MutexGuard;

    fn assert_send_and_sync()
    where
        ChunksMut<'static, Cell<i32>>: Send,
        ChunksMut<'static, MutexGuard<'static, u32>>: Sync,
        ChunksExactMut<'static, Cell<i32>>: Send,
        ChunksExactMut<'static, MutexGuard<'static, u32>>: Sync,
        RChunksMut<'static, Cell<i32>>: Send,
        RChunksMut<'static, MutexGuard<'static, u32>>: Sync,
        RChunksExactMut<'static, Cell<i32>>: Send,
        RChunksExactMut<'static, MutexGuard<'static, u32>>: Sync,
    {
    }

    assert_send_and_sync();
}

#[test]
fn test_windows_count() {
    let v: &[i32] = &[0, 1, 2, 3, 4, 5];
    let c = v.windows(3);
    assert_eq!(c.count(), 4);

    let v2: &[i32] = &[0, 1, 2, 3, 4];
    let c2 = v2.windows(6);
    assert_eq!(c2.count(), 0);

    let v3: &[i32] = &[];
    let c3 = v3.windows(2);
    assert_eq!(c3.count(), 0);

    let v4 = &[(); usize::MAX];
    let c4 = v4.windows(1);
    assert_eq!(c4.count(), usize::MAX);
}

#[test]
fn test_windows_nth() {
    let v: &[i32] = &[0, 1, 2, 3, 4, 5];
    let mut c = v.windows(2);
    assert_eq!(c.nth(2).unwrap()[1], 3);
    assert_eq!(c.next().unwrap()[0], 3);

    let v2: &[i32] = &[0, 1, 2, 3, 4];
    let mut c2 = v2.windows(4);
    assert_eq!(c2.nth(1).unwrap()[1], 2);
    assert_eq!(c2.next(), None);
}

#[test]
fn test_windows_nth_back() {
    let v: &[i32] = &[0, 1, 2, 3, 4, 5];
    let mut c = v.windows(2);
    assert_eq!(c.nth_back(2).unwrap()[0], 2);
    assert_eq!(c.next_back().unwrap()[1], 2);

    let v2: &[i32] = &[0, 1, 2, 3, 4];
    let mut c2 = v2.windows(4);
    assert_eq!(c2.nth_back(1).unwrap()[1], 1);
    assert_eq!(c2.next_back(), None);
}

#[test]
fn test_windows_last() {
    let v: &[i32] = &[0, 1, 2, 3, 4, 5];
    let c = v.windows(2);
    assert_eq!(c.last().unwrap()[1], 5);

    let v2: &[i32] = &[0, 1, 2, 3, 4];
    let c2 = v2.windows(2);
    assert_eq!(c2.last().unwrap()[0], 3);
}

#[test]
fn test_windows_zip() {
    let v1: &[i32] = &[0, 1, 2, 3, 4];
    let v2: &[i32] = &[6, 7, 8, 9, 10];

    let res = v1
        .windows(2)
        .zip(v2.windows(2))
        .map(|(a, b)| a.iter().sum::<i32>() + b.iter().sum::<i32>())
        .collect::<Vec<_>>();

    assert_eq!(res, [14, 18, 22, 26]);
}

#[test]
fn test_iter_ref_consistency() {
    use std::fmt::Debug;

    fn test<T: Copy + Debug + PartialEq>(x: T) {
        let v: &[T] = &[x, x, x];
        let v_ptrs: [*const T; 3] = match v {
            [v1, v2, v3] => [v1 as *const _, v2 as *const _, v3 as *const _],
            _ => unreachable!(),
        };
        let len = v.len();

        // nth(i)
        for i in 0..len {
            assert_eq!(&v[i] as *const _, v_ptrs[i]); // check the v_ptrs array, just to be sure
            let nth = v.iter().nth(i).unwrap();
            assert_eq!(nth as *const _, v_ptrs[i]);
        }
        assert_eq!(v.iter().nth(len), None, "nth(len) should return None");

        // stepping through with nth(0)
        {
            let mut it = v.iter();
            for i in 0..len {
                let next = it.nth(0).unwrap();
                assert_eq!(next as *const _, v_ptrs[i]);
            }
            assert_eq!(it.nth(0), None);
        }

        // next()
        {
            let mut it = v.iter();
            for i in 0..len {
                let remaining = len - i;
                assert_eq!(it.size_hint(), (remaining, Some(remaining)));

                let next = it.next().unwrap();
                assert_eq!(next as *const _, v_ptrs[i]);
            }
            assert_eq!(it.size_hint(), (0, Some(0)));
            assert_eq!(it.next(), None, "The final call to next() should return None");
        }

        // next_back()
        {
            let mut it = v.iter();
            for i in 0..len {
                let remaining = len - i;
                assert_eq!(it.size_hint(), (remaining, Some(remaining)));

                let prev = it.next_back().unwrap();
                assert_eq!(prev as *const _, v_ptrs[remaining - 1]);
            }
            assert_eq!(it.size_hint(), (0, Some(0)));
            assert_eq!(it.next_back(), None, "The final call to next_back() should return None");
        }
    }

    fn test_mut<T: Copy + Debug + PartialEq>(x: T) {
        let v: &mut [T] = &mut [x, x, x];
        let v_ptrs: [*mut T; 3] = match v {
            &mut [ref v1, ref v2, ref v3] => {
                [v1 as *const _ as *mut _, v2 as *const _ as *mut _, v3 as *const _ as *mut _]
            }
            _ => unreachable!(),
        };
        let len = v.len();

        // nth(i)
        for i in 0..len {
            assert_eq!(&mut v[i] as *mut _, v_ptrs[i]); // check the v_ptrs array, just to be sure
            let nth = v.iter_mut().nth(i).unwrap();
            assert_eq!(nth as *mut _, v_ptrs[i]);
        }
        assert_eq!(v.iter().nth(len), None, "nth(len) should return None");

        // stepping through with nth(0)
        {
            let mut it = v.iter();
            for i in 0..len {
                let next = it.nth(0).unwrap();
                assert_eq!(next as *const _, v_ptrs[i]);
            }
            assert_eq!(it.nth(0), None);
        }

        // next()
        {
            let mut it = v.iter_mut();
            for i in 0..len {
                let remaining = len - i;
                assert_eq!(it.size_hint(), (remaining, Some(remaining)));

                let next = it.next().unwrap();
                assert_eq!(next as *mut _, v_ptrs[i]);
            }
            assert_eq!(it.size_hint(), (0, Some(0)));
            assert_eq!(it.next(), None, "The final call to next() should return None");
        }

        // next_back()
        {
            let mut it = v.iter_mut();
            for i in 0..len {
                let remaining = len - i;
                assert_eq!(it.size_hint(), (remaining, Some(remaining)));

                let prev = it.next_back().unwrap();
                assert_eq!(prev as *mut _, v_ptrs[remaining - 1]);
            }
            assert_eq!(it.size_hint(), (0, Some(0)));
            assert_eq!(it.next_back(), None, "The final call to next_back() should return None");
        }
    }

    // Make sure iterators and slice patterns yield consistent addresses for various types,
    // including ZSTs.
    test(0u32);
    test(());
    test([0u32; 0]); // ZST with alignment > 0
    test_mut(0u32);
    test_mut(());
    test_mut([0u32; 0]); // ZST with alignment > 0
}

// The current implementation of SliceIndex fails to handle methods
// orthogonally from range types; therefore, it is worth testing
// all of the indexing operations on each input.
mod slice_index {
    // This checks all six indexing methods, given an input range that
    // should succeed. (it is NOT suitable for testing invalid inputs)
    macro_rules! assert_range_eq {
        ($arr:expr, $range:expr, $expected:expr) => {
            let mut arr = $arr;
            let mut expected = $expected;
            {
                let s: &[_] = &arr;
                let expected: &[_] = &expected;

                assert_eq!(&s[$range], expected, "(in assertion for: index)");
                assert_eq!(s.get($range), Some(expected), "(in assertion for: get)");
                unsafe {
                    assert_eq!(
                        s.get_unchecked($range),
                        expected,
                        "(in assertion for: get_unchecked)",
                    );
                }
            }
            {
                let s: &mut [_] = &mut arr;
                let expected: &mut [_] = &mut expected;

                assert_eq!(&mut s[$range], expected, "(in assertion for: index_mut)",);
                assert_eq!(
                    s.get_mut($range),
                    Some(&mut expected[..]),
                    "(in assertion for: get_mut)",
                );
                unsafe {
                    assert_eq!(
                        s.get_unchecked_mut($range),
                        expected,
                        "(in assertion for: get_unchecked_mut)",
                    );
                }
            }
        };
    }

    // Make sure the macro can actually detect bugs,
    // because if it can't, then what are we even doing here?
    //
    // (Be aware this only demonstrates the ability to detect bugs
    //  in the FIRST method that panics, as the macro is not designed
    //  to be used in `should_panic`)
    #[test]
    #[should_panic(expected = "out of range")]
    fn assert_range_eq_can_fail_by_panic() {
        assert_range_eq!([0, 1, 2], 0..5, [0, 1, 2]);
    }

    // (Be aware this only demonstrates the ability to detect bugs
    //  in the FIRST method it calls, as the macro is not designed
    //  to be used in `should_panic`)
    #[test]
    #[should_panic(expected = "==")]
    fn assert_range_eq_can_fail_by_inequality() {
        assert_range_eq!([0, 1, 2], 0..2, [0, 1, 2]);
    }

    // Test cases for bad index operations.
    //
    // This generates `should_panic` test cases for Index/IndexMut
    // and `None` test cases for get/get_mut.
    macro_rules! panic_cases {
        ($(
            // each test case needs a unique name to namespace the tests
            in mod $case_name:ident {
                data: $data:expr;

                // optional:
                //
                // one or more similar inputs for which data[input] succeeds,
                // and the corresponding output as an array. This helps validate
                // "critical points" where an input range straddles the boundary
                // between valid and invalid.
                // (such as the input `len..len`, which is just barely valid)
                $(
                    good: data[$good:expr] == $output:expr;
                )*

                bad: data[$bad:expr];
                message: $expect_msg:expr;
            }
        )*) => {$(
            mod $case_name {
                #[allow(unused_imports)]
                use core::ops::Bound;

                #[test]
                fn pass() {
                    let mut v = $data;

                    $( assert_range_eq!($data, $good, $output); )*

                    {
                        let v: &[_] = &v;
                        assert_eq!(v.get($bad), None, "(in None assertion for get)");
                    }

                    {
                        let v: &mut [_] = &mut v;
                        assert_eq!(v.get_mut($bad), None, "(in None assertion for get_mut)");
                    }
                }

                #[test]
                #[should_panic(expected = $expect_msg)]
                fn index_fail() {
                    let v = $data;
                    let v: &[_] = &v;
                    let _v = &v[$bad];
                }

                #[test]
                #[should_panic(expected = $expect_msg)]
                fn index_mut_fail() {
                    let mut v = $data;
                    let v: &mut [_] = &mut v;
                    let _v = &mut v[$bad];
                }
            }
        )*};
    }

    #[test]
    fn simple() {
        let v = [0, 1, 2, 3, 4, 5];

        assert_range_eq!(v, .., [0, 1, 2, 3, 4, 5]);
        assert_range_eq!(v, ..2, [0, 1]);
        assert_range_eq!(v, ..=1, [0, 1]);
        assert_range_eq!(v, 2.., [2, 3, 4, 5]);
        assert_range_eq!(v, 1..4, [1, 2, 3]);
        assert_range_eq!(v, 1..=3, [1, 2, 3]);
    }

    panic_cases! {
        in mod rangefrom_len {
            data: [0, 1, 2, 3, 4, 5];

            good: data[6..] == [];
            bad: data[7..];
            message: "out of range";
        }

        in mod rangeto_len {
            data: [0, 1, 2, 3, 4, 5];

            good: data[..6] == [0, 1, 2, 3, 4, 5];
            bad: data[..7];
            message: "out of range";
        }

        in mod rangetoinclusive_len {
            data: [0, 1, 2, 3, 4, 5];

            good: data[..=5] == [0, 1, 2, 3, 4, 5];
            bad: data[..=6];
            message: "out of range";
        }

        in mod rangeinclusive_len {
            data: [0, 1, 2, 3, 4, 5];

            good: data[0..=5] == [0, 1, 2, 3, 4, 5];
            bad: data[0..=6];
            message: "out of range";
        }

        in mod range_len_len {
            data: [0, 1, 2, 3, 4, 5];

            good: data[6..6] == [];
            bad: data[7..7];
            message: "out of range";
        }

        in mod rangeinclusive_len_len {
            data: [0, 1, 2, 3, 4, 5];

            good: data[6..=5] == [];
            bad: data[7..=6];
            message: "out of range";
        }

        in mod boundpair_len {
            data: [0, 1, 2, 3, 4, 5];

            good: data[(Bound::Included(6), Bound::Unbounded)] == [];
            good: data[(Bound::Unbounded, Bound::Included(5))] == [0, 1, 2, 3, 4, 5];
            good: data[(Bound::Unbounded, Bound::Excluded(6))] == [0, 1, 2, 3, 4, 5];
            good: data[(Bound::Included(0), Bound::Included(5))] == [0, 1, 2, 3, 4, 5];
            good: data[(Bound::Included(0), Bound::Excluded(6))] == [0, 1, 2, 3, 4, 5];
            good: data[(Bound::Included(2), Bound::Excluded(4))] == [2, 3];
            good: data[(Bound::Excluded(1), Bound::Included(4))] == [2, 3, 4];
            good: data[(Bound::Excluded(5), Bound::Excluded(6))] == [];
            good: data[(Bound::Included(6), Bound::Excluded(6))] == [];
            good: data[(Bound::Excluded(5), Bound::Included(5))] == [];
            good: data[(Bound::Included(6), Bound::Included(5))] == [];
            bad: data[(Bound::Unbounded, Bound::Included(6))];
            message: "out of range";
        }
    }

    panic_cases! {
        in mod rangeinclusive_exhausted {
            data: [0, 1, 2, 3, 4, 5];

            good: data[0..=5] == [0, 1, 2, 3, 4, 5];
            good: data[{
                let mut iter = 0..=5;
                iter.by_ref().count(); // exhaust it
                iter
            }] == [];

            // 0..=6 is out of range before exhaustion, so it
            // stands to reason that it still would be after.
            bad: data[{
                let mut iter = 0..=6;
                iter.by_ref().count(); // exhaust it
                iter
            }];
            message: "out of range";
        }
    }

    panic_cases! {
        in mod range_neg_width {
            data: [0, 1, 2, 3, 4, 5];

            good: data[4..4] == [];
            bad: data[4..3];
            message: "but ends at";
        }

        in mod rangeinclusive_neg_width {
            data: [0, 1, 2, 3, 4, 5];

            good: data[4..=3] == [];
            bad: data[4..=2];
            message: "but ends at";
        }

        in mod boundpair_neg_width {
            data: [0, 1, 2, 3, 4, 5];

            good: data[(Bound::Included(4), Bound::Excluded(4))] == [];
            bad: data[(Bound::Included(4), Bound::Excluded(3))];
            message: "but ends at";
        }
    }

    panic_cases! {
        in mod rangeinclusive_overflow {
            data: [0, 1];

            // note: using 0 specifically ensures that the result of overflowing is 0..0,
            //       so that `get` doesn't simply return None for the wrong reason.
            bad: data[0 ..= usize::MAX];
            message: "out of range";
        }

        in mod rangetoinclusive_overflow {
            data: [0, 1];

            bad: data[..= usize::MAX];
            message: "out of range";
        }

        in mod boundpair_overflow_end {
            data: [0; 1];

            bad: data[(Bound::Unbounded, Bound::Included(usize::MAX))];
            message: "out of range";
        }

        in mod boundpair_overflow_start {
            data: [0; 1];

            bad: data[(Bound::Excluded(usize::MAX), Bound::Unbounded)];
            message: "out of range";
        }
    } // panic_cases!
}

#[test]
fn test_find_rfind() {
    let v = [0, 1, 2, 3, 4, 5];
    let mut iter = v.iter();
    let mut i = v.len();
    while let Some(&elt) = iter.rfind(|_| true) {
        i -= 1;
        assert_eq!(elt, v[i]);
    }
    assert_eq!(i, 0);
    assert_eq!(v.iter().rfind(|&&x| x <= 3), Some(&3));
}

#[test]
fn test_iter_folds() {
    let a = [1, 2, 3, 4, 5]; // len>4 so the unroll is used
    assert_eq!(a.iter().fold(0, |acc, &x| 2 * acc + x), 57);
    assert_eq!(a.iter().rfold(0, |acc, &x| 2 * acc + x), 129);
    let fold = |acc: i32, &x| acc.checked_mul(2)?.checked_add(x);
    assert_eq!(a.iter().try_fold(0, &fold), Some(57));
    assert_eq!(a.iter().try_rfold(0, &fold), Some(129));

    // short-circuiting try_fold, through other methods
    let a = [0, 1, 2, 3, 5, 5, 5, 7, 8, 9];
    let mut iter = a.iter();
    assert_eq!(iter.position(|&x| x == 3), Some(3));
    assert_eq!(iter.rfind(|&&x| x == 5), Some(&5));
    assert_eq!(iter.len(), 2);
}

#[test]
fn test_rotate_left() {
    const N: usize = 600;
    let a: &mut [_] = &mut [0; N];
    for i in 0..N {
        a[i] = i;
    }

    a.rotate_left(42);
    let k = N - 42;

    for i in 0..N {
        assert_eq!(a[(i + k) % N], i);
    }
}

#[test]
fn test_rotate_right() {
    const N: usize = 600;
    let a: &mut [_] = &mut [0; N];
    for i in 0..N {
        a[i] = i;
    }

    a.rotate_right(42);

    for i in 0..N {
        assert_eq!(a[(i + 42) % N], i);
    }
}

#[test]
#[cfg_attr(miri, ignore)] // Miri is too slow
fn brute_force_rotate_test_0() {
    // In case of edge cases involving multiple algorithms
    let n = 300;
    for len in 0..n {
        for s in 0..len {
            let mut v = Vec::with_capacity(len);
            for i in 0..len {
                v.push(i);
            }
            v[..].rotate_right(s);
            for i in 0..v.len() {
                assert_eq!(v[i], v.len().wrapping_add(i.wrapping_sub(s)) % v.len());
            }
        }
    }
}

#[test]
fn brute_force_rotate_test_1() {
    // `ptr_rotate` covers so many kinds of pointer usage, that this is just a good test for
    // pointers in general. This uses a `[usize; 4]` to hit all algorithms without overwhelming miri
    let n = 30;
    for len in 0..n {
        for s in 0..len {
            let mut v: Vec<[usize; 4]> = Vec::with_capacity(len);
            for i in 0..len {
                v.push([i, 0, 0, 0]);
            }
            v[..].rotate_right(s);
            for i in 0..v.len() {
                assert_eq!(v[i][0], v.len().wrapping_add(i.wrapping_sub(s)) % v.len());
            }
        }
    }
}

#[test]
#[cfg(not(target_arch = "wasm32"))]
#[cfg_attr(miri, ignore)] // Miri is too slow
fn select_nth_unstable() {
    use core::cmp::Ordering::{Equal, Greater, Less};

    use rand::Rng;
    use rand::seq::IndexedRandom;

    let mut rng = crate::test_rng();

    for len in (2..21).chain(500..501) {
        let mut orig = vec![0; len];

        for &modulus in &[5, 10, 1000] {
            for _ in 0..10 {
                for i in 0..len {
                    orig[i] = rng.random::<i32>() % modulus;
                }

                let v_sorted = {
                    let mut v = orig.clone();
                    v.sort();
                    v
                };

                // Sort in default order.
                for pivot in 0..len {
                    let mut v = orig.clone();
                    v.select_nth_unstable(pivot);

                    assert_eq!(v_sorted[pivot], v[pivot]);
                    for i in 0..pivot {
                        for j in pivot..len {
                            assert!(v[i] <= v[j]);
                        }
                    }
                }

                // Sort in ascending order.
                for pivot in 0..len {
                    let mut v = orig.clone();
                    let (left, pivot, right) = v.select_nth_unstable_by(pivot, |a, b| a.cmp(b));

                    assert_eq!(left.len() + right.len(), len - 1);

                    for l in left {
                        assert!(l <= pivot);
                        for r in right.iter_mut() {
                            assert!(l <= r);
                            assert!(pivot <= r);
                        }
                    }
                }

                // Sort in descending order.
                let sort_descending_comparator = |a: &i32, b: &i32| b.cmp(a);
                let v_sorted_descending = {
                    let mut v = orig.clone();
                    v.sort_by(sort_descending_comparator);
                    v
                };

                for pivot in 0..len {
                    let mut v = orig.clone();
                    v.select_nth_unstable_by(pivot, sort_descending_comparator);

                    assert_eq!(v_sorted_descending[pivot], v[pivot]);
                    for i in 0..pivot {
                        for j in pivot..len {
                            assert!(v[j] <= v[i]);
                        }
                    }
                }
            }
        }
    }

    // Sort at index using a completely random comparison function.
    // This will reorder the elements *somehow*, but won't panic.
    let mut v = [0; 500];
    for i in 0..v.len() {
        v[i] = i as i32;
    }

    for pivot in 0..v.len() {
        v.select_nth_unstable_by(pivot, |_, _| *[Less, Equal, Greater].choose(&mut rng).unwrap());
        v.sort();
        for i in 0..v.len() {
            assert_eq!(v[i], i as i32);
        }
    }

    // Should not panic.
    [(); 10].select_nth_unstable(0);
    [(); 10].select_nth_unstable(5);
    [(); 10].select_nth_unstable(9);
    [(); 100].select_nth_unstable(0);
    [(); 100].select_nth_unstable(50);
    [(); 100].select_nth_unstable(99);

    let mut v = [0xDEADBEEFu64];
    v.select_nth_unstable(0);
    assert!(v == [0xDEADBEEF]);
}

#[test]
#[should_panic(expected = "index 0 greater than length of slice")]
fn select_nth_unstable_zero_length() {
    [0i32; 0].select_nth_unstable(0);
}

#[test]
#[should_panic(expected = "index 20 greater than length of slice")]
fn select_nth_unstable_past_length() {
    [0i32; 10].select_nth_unstable(20);
}

pub mod memchr {
    use core::slice::memchr::{memchr, memrchr};

    // test fallback implementations on all platforms
    #[test]
    fn matches_one() {
        assert_eq!(Some(0), memchr(b'a', b"a"));
    }

    #[test]
    fn matches_begin() {
        assert_eq!(Some(0), memchr(b'a', b"aaaa"));
    }

    #[test]
    fn matches_end() {
        assert_eq!(Some(4), memchr(b'z', b"aaaaz"));
    }

    #[test]
    fn matches_nul() {
        assert_eq!(Some(4), memchr(b'\x00', b"aaaa\x00"));
    }

    #[test]
    fn matches_past_nul() {
        assert_eq!(Some(5), memchr(b'z', b"aaaa\x00z"));
    }

    #[test]
    fn no_match_empty() {
        assert_eq!(None, memchr(b'a', b""));
    }

    #[test]
    fn no_match() {
        assert_eq!(None, memchr(b'a', b"xyz"));
    }

    #[test]
    fn matches_one_reversed() {
        assert_eq!(Some(0), memrchr(b'a', b"a"));
    }

    #[test]
    fn matches_begin_reversed() {
        assert_eq!(Some(3), memrchr(b'a', b"aaaa"));
    }

    #[test]
    fn matches_end_reversed() {
        assert_eq!(Some(0), memrchr(b'z', b"zaaaa"));
    }

    #[test]
    fn matches_nul_reversed() {
        assert_eq!(Some(4), memrchr(b'\x00', b"aaaa\x00"));
    }

    #[test]
    fn matches_past_nul_reversed() {
        assert_eq!(Some(0), memrchr(b'z', b"z\x00aaaa"));
    }

    #[test]
    fn no_match_empty_reversed() {
        assert_eq!(None, memrchr(b'a', b""));
    }

    #[test]
    fn no_match_reversed() {
        assert_eq!(None, memrchr(b'a', b"xyz"));
    }

    #[test]
    fn each_alignment_reversed() {
        let mut data = [1u8; 64];
        let needle = 2;
        let pos = 40;
        data[pos] = needle;
        for start in 0..16 {
            assert_eq!(Some(pos - start), memrchr(needle, &data[start..]));
        }
    }
}

#[test]
fn test_align_to_simple() {
    let bytes = [1u8, 2, 3, 4, 5, 6, 7];
    let (prefix, aligned, suffix) = unsafe { bytes.align_to::<u16>() };
    assert_eq!(aligned.len(), 3);
    assert!(prefix == [1] || suffix == [7]);
    let expect1 = [1 << 8 | 2, 3 << 8 | 4, 5 << 8 | 6];
    let expect2 = [1 | 2 << 8, 3 | 4 << 8, 5 | 6 << 8];
    let expect3 = [2 << 8 | 3, 4 << 8 | 5, 6 << 8 | 7];
    let expect4 = [2 | 3 << 8, 4 | 5 << 8, 6 | 7 << 8];
    assert!(
        aligned == expect1 || aligned == expect2 || aligned == expect3 || aligned == expect4,
        "aligned={:?} expected={:?} || {:?} || {:?} || {:?}",
        aligned,
        expect1,
        expect2,
        expect3,
        expect4
    );
}

#[test]
fn test_align_to_zst() {
    let bytes = [1, 2, 3, 4, 5, 6, 7];
    let (prefix, aligned, suffix) = unsafe { bytes.align_to::<()>() };
    assert_eq!(aligned.len(), 0);
    assert!(prefix == [1, 2, 3, 4, 5, 6, 7] || suffix == [1, 2, 3, 4, 5, 6, 7]);
}

#[test]
fn test_align_to_non_trivial() {
    #[repr(align(8))]
    struct U64(#[allow(dead_code)] u64, #[allow(dead_code)] u64);
    #[repr(align(8))]
    struct U64U64U32(#[allow(dead_code)] u64, #[allow(dead_code)] u64, #[allow(dead_code)] u32);
    let data = [
        U64(1, 2),
        U64(3, 4),
        U64(5, 6),
        U64(7, 8),
        U64(9, 10),
        U64(11, 12),
        U64(13, 14),
        U64(15, 16),
    ];
    let (prefix, aligned, suffix) = unsafe { data.align_to::<U64U64U32>() };
    assert_eq!(aligned.len(), 4);
    assert_eq!(prefix.len() + suffix.len(), 2);
}

#[test]
fn test_align_to_empty_mid() {
    // Make sure that we do not create empty unaligned slices for the mid part, even when the
    // overall slice is too short to contain an aligned address.
    let bytes = [1, 2, 3, 4, 5, 6, 7];
    type Chunk = u32;
    for offset in 0..4 {
        let (_, mid, _) = unsafe { bytes[offset..offset + 1].align_to::<Chunk>() };
        assert_eq!(mid.as_ptr() as usize % align_of::<Chunk>(), 0);
    }
}

#[test]
fn test_align_to_mut_aliasing() {
    let mut val = [1u8, 2, 3, 4, 5];
    // `align_to_mut` used to create `mid` in a way that there was some intermediate
    // incorrect aliasing, invalidating the resulting `mid` slice.
    let (begin, mid, end) = unsafe { val.align_to_mut::<[u8; 2]>() };
    assert!(begin.len() == 0);
    assert!(end.len() == 1);
    mid[0] = mid[1];
    assert_eq!(val, [3, 4, 3, 4, 5])
}

#[test]
fn test_slice_partition_dedup_by() {
    let mut slice: [i32; 9] = [1, -1, 2, 3, 1, -5, 5, -2, 2];

    let (dedup, duplicates) = slice.partition_dedup_by(|a, b| a.abs() == b.abs());

    assert_eq!(dedup, [1, 2, 3, 1, -5, -2]);
    assert_eq!(duplicates, [5, -1, 2]);
}

#[test]
fn test_slice_partition_dedup_empty() {
    let mut slice: [i32; 0] = [];

    let (dedup, duplicates) = slice.partition_dedup();

    assert_eq!(dedup, []);
    assert_eq!(duplicates, []);
}

#[test]
fn test_slice_partition_dedup_one() {
    let mut slice = [12];

    let (dedup, duplicates) = slice.partition_dedup();

    assert_eq!(dedup, [12]);
    assert_eq!(duplicates, []);
}

#[test]
fn test_slice_partition_dedup_multiple_ident() {
    let mut slice = [12, 12, 12, 12, 12, 11, 11, 11, 11, 11, 11];

    let (dedup, duplicates) = slice.partition_dedup();

    assert_eq!(dedup, [12, 11]);
    assert_eq!(duplicates, [12, 12, 12, 12, 11, 11, 11, 11, 11]);
}

#[test]
fn test_slice_partition_dedup_partialeq() {
    #[derive(Debug)]
    struct Foo(i32, #[allow(dead_code)] i32);

    impl PartialEq for Foo {
        fn eq(&self, other: &Foo) -> bool {
            self.0 == other.0
        }
    }

    let mut slice = [Foo(0, 1), Foo(0, 5), Foo(1, 7), Foo(1, 9)];

    let (dedup, duplicates) = slice.partition_dedup();

    assert_eq!(dedup, [Foo(0, 1), Foo(1, 7)]);
    assert_eq!(duplicates, [Foo(0, 5), Foo(1, 9)]);
}

#[test]
fn test_copy_within() {
    // Start to end, with a RangeTo.
    let mut bytes = *b"Hello, World!";
    bytes.copy_within(..3, 10);
    assert_eq!(&bytes, b"Hello, WorHel");

    // End to start, with a RangeFrom.
    let mut bytes = *b"Hello, World!";
    bytes.copy_within(10.., 0);
    assert_eq!(&bytes, b"ld!lo, World!");

    // Overlapping, with a RangeInclusive.
    let mut bytes = *b"Hello, World!";
    bytes.copy_within(0..=11, 1);
    assert_eq!(&bytes, b"HHello, World");

    // Whole slice, with a RangeFull.
    let mut bytes = *b"Hello, World!";
    bytes.copy_within(.., 0);
    assert_eq!(&bytes, b"Hello, World!");

    // Ensure that copying at the end of slice won't cause UB.
    let mut bytes = *b"Hello, World!";
    bytes.copy_within(13..13, 5);
    assert_eq!(&bytes, b"Hello, World!");
    bytes.copy_within(5..5, 13);
    assert_eq!(&bytes, b"Hello, World!");
}

#[test]
#[should_panic(expected = "range end index 14 out of range for slice of length 13")]
fn test_copy_within_panics_src_too_long() {
    let mut bytes = *b"Hello, World!";
    // The length is only 13, so 14 is out of bounds.
    bytes.copy_within(10..14, 0);
}

#[test]
#[should_panic(expected = "dest is out of bounds")]
fn test_copy_within_panics_dest_too_long() {
    let mut bytes = *b"Hello, World!";
    // The length is only 13, so a slice of length 4 starting at index 10 is out of bounds.
    bytes.copy_within(0..4, 10);
}

#[test]
#[should_panic(expected = "slice index starts at 2 but ends at 1")]
fn test_copy_within_panics_src_inverted() {
    let mut bytes = *b"Hello, World!";
    // 2 is greater than 1, so this range is invalid.
    bytes.copy_within(2..1, 0);
}
#[test]
#[should_panic(expected = "out of range")]
fn test_copy_within_panics_src_out_of_bounds() {
    let mut bytes = *b"Hello, World!";
    // an inclusive range ending at usize::MAX would make src_end overflow
    bytes.copy_within(usize::MAX..=usize::MAX, 0);
}

#[test]
fn test_is_sorted() {
    let empty: [i32; 0] = [];

    // Tests on integers
    assert!([1, 2, 2, 9].is_sorted());
    assert!(![1, 3, 2].is_sorted());
    assert!([0].is_sorted());
    assert!([0, 0].is_sorted());
    assert!(empty.is_sorted());

    // Tests on floats
    assert!([1.0f32, 2.0, 2.0, 9.0].is_sorted());
    assert!(![1.0f32, 3.0f32, 2.0f32].is_sorted());
    assert!([0.0f32].is_sorted());
    assert!([0.0f32, 0.0f32].is_sorted());
    // Test cases with NaNs
    assert!([f32::NAN].is_sorted());
    assert!(![f32::NAN, f32::NAN].is_sorted());
    assert!(![0.0, 1.0, f32::NAN].is_sorted());
    // Tests from <https://github.com/rust-lang/rust/pull/55045#discussion_r229689884>
    assert!(![f32::NAN, f32::NAN, f32::NAN].is_sorted());
    assert!(![1.0, f32::NAN, 2.0].is_sorted());
    assert!(![2.0, f32::NAN, 1.0].is_sorted());
    assert!(![2.0, f32::NAN, 1.0, 7.0].is_sorted());
    assert!(![2.0, f32::NAN, 1.0, 0.0].is_sorted());
    assert!(![-f32::NAN, -1.0, 0.0, 1.0, f32::NAN].is_sorted());
    assert!(![f32::NAN, -f32::NAN, -1.0, 0.0, 1.0].is_sorted());
    assert!(![1.0, f32::NAN, -f32::NAN, -1.0, 0.0].is_sorted());
    assert!(![0.0, 1.0, f32::NAN, -f32::NAN, -1.0].is_sorted());
    assert!(![-1.0, 0.0, 1.0, f32::NAN, -f32::NAN].is_sorted());

    // Tests for is_sorted_by
    assert!(![6, 2, 8, 5, 1, -60, 1337].is_sorted());
    assert!([6, 2, 8, 5, 1, -60, 1337].is_sorted_by(|_, _| true));

    // Tests for is_sorted_by_key
    assert!([-2, -1, 0, 3].is_sorted());
    assert!(![-2i32, -1, 0, 3].is_sorted_by_key(|n| n.abs()));
    assert!(!["c", "bb", "aaa"].is_sorted());
    assert!(["c", "bb", "aaa"].is_sorted_by_key(|s| s.len()));
}

#[test]
fn test_slice_run_destructors() {
    // Make sure that destructors get run on slice literals
    struct Foo<'a> {
        x: &'a Cell<isize>,
    }

    impl<'a> Drop for Foo<'a> {
        fn drop(&mut self) {
            self.x.set(self.x.get() + 1);
        }
    }

    fn foo(x: &Cell<isize>) -> Foo<'_> {
        Foo { x }
    }

    let x = &Cell::new(0);

    {
        let l = &[foo(x)];
        assert_eq!(l[0].x.get(), 0);
    }

    assert_eq!(x.get(), 1);
}

#[test]
fn test_const_from_ref() {
    const VALUE: &i32 = &1;
    const SLICE: &[i32] = core::slice::from_ref(VALUE);

    assert!(core::ptr::eq(VALUE, &SLICE[0]))
}

#[test]
fn test_slice_fill_with_uninit() {
    // This should not UB. See #87891
    let mut a = [MaybeUninit::<u8>::uninit(); 10];
    a.fill(MaybeUninit::uninit());
}

#[test]
fn test_swap() {
    let mut x = ["a", "b", "c", "d"];
    x.swap(1, 3);
    assert_eq!(x, ["a", "d", "c", "b"]);
    x.swap(0, 3);
    assert_eq!(x, ["b", "d", "c", "a"]);
}

mod swap_panics {
    #[test]
    #[should_panic(expected = "index out of bounds: the len is 4 but the index is 4")]
    fn index_a_equals_len() {
        let mut x = ["a", "b", "c", "d"];
        x.swap(4, 2);
    }

    #[test]
    #[should_panic(expected = "index out of bounds: the len is 4 but the index is 4")]
    fn index_b_equals_len() {
        let mut x = ["a", "b", "c", "d"];
        x.swap(2, 4);
    }

    #[test]
    #[should_panic(expected = "index out of bounds: the len is 4 but the index is 5")]
    fn index_a_greater_than_len() {
        let mut x = ["a", "b", "c", "d"];
        x.swap(5, 2);
    }

    #[test]
    #[should_panic(expected = "index out of bounds: the len is 4 but the index is 5")]
    fn index_b_greater_than_len() {
        let mut x = ["a", "b", "c", "d"];
        x.swap(2, 5);
    }
}

#[test]
fn slice_split_first_chunk_mut() {
    let v = &mut [1, 2, 3, 4, 5, 6][..];

    {
        let (left, right) = v.split_first_chunk_mut::<0>().unwrap();
        assert_eq!(left, &mut []);
        assert_eq!(right, [1, 2, 3, 4, 5, 6]);
    }

    {
        let (left, right) = v.split_first_chunk_mut::<6>().unwrap();
        assert_eq!(left, &mut [1, 2, 3, 4, 5, 6]);
        assert_eq!(right, []);
    }

    {
        assert!(v.split_first_chunk_mut::<7>().is_none());
    }
}

#[test]
fn slice_split_last_chunk_mut() {
    let v = &mut [1, 2, 3, 4, 5, 6][..];

    {
        let (left, right) = v.split_last_chunk_mut::<0>().unwrap();
        assert_eq!(left, [1, 2, 3, 4, 5, 6]);
        assert_eq!(right, &mut []);
    }

    {
        let (left, right) = v.split_last_chunk_mut::<6>().unwrap();
        assert_eq!(left, []);
        assert_eq!(right, &mut [1, 2, 3, 4, 5, 6]);
    }

    {
        assert!(v.split_last_chunk_mut::<7>().is_none());
    }
}

#[test]
fn split_as_slice() {
    let arr = [1, 2, 3, 4, 5, 6];
    let mut split = arr.split(|v| v % 2 == 0);
    assert_eq!(split.as_slice(), &[1, 2, 3, 4, 5, 6]);
    assert!(split.next().is_some());
    assert_eq!(split.as_slice(), &[3, 4, 5, 6]);
    assert!(split.next().is_some());
    assert!(split.next().is_some());
    assert_eq!(split.as_slice(), &[]);
}

#[test]
fn slice_split_once() {
    let v = &[1, 2, 3, 2, 4][..];

    assert_eq!(v.split_once(|&x| x == 2), Some((&[1][..], &[3, 2, 4][..])));
    assert_eq!(v.split_once(|&x| x == 1), Some((&[][..], &[2, 3, 2, 4][..])));
    assert_eq!(v.split_once(|&x| x == 4), Some((&[1, 2, 3, 2][..], &[][..])));
    assert_eq!(v.split_once(|&x| x == 0), None);
}

#[test]
fn slice_rsplit_once() {
    let v = &[1, 2, 3, 2, 4][..];

    assert_eq!(v.rsplit_once(|&x| x == 2), Some((&[1, 2, 3][..], &[4][..])));
    assert_eq!(v.rsplit_once(|&x| x == 1), Some((&[][..], &[2, 3, 2, 4][..])));
    assert_eq!(v.rsplit_once(|&x| x == 4), Some((&[1, 2, 3, 2][..], &[][..])));
    assert_eq!(v.rsplit_once(|&x| x == 0), None);
}

macro_rules! split_off_tests {
    (slice: &[], $($tts:tt)*) => {
        split_off_tests!(ty: &[()], slice: &[], $($tts)*);
    };
    (slice: &mut [], $($tts:tt)*) => {
        split_off_tests!(ty: &mut [()], slice: &mut [], $($tts)*);
    };
    (slice: &$slice:expr, $($tts:tt)*) => {
        split_off_tests!(ty: &[_], slice: &$slice, $($tts)*);
    };
    (slice: &mut $slice:expr, $($tts:tt)*) => {
        split_off_tests!(ty: &mut [_], slice: &mut $slice, $($tts)*);
    };
    (ty: $ty:ty, slice: $slice:expr, method: $method:ident, $(($test_name:ident, ($($args:expr),*), $output:expr, $remaining:expr),)*) => {
        $(
            #[test]
            fn $test_name() {
                let mut slice: $ty = $slice;
                assert_eq!($output, slice.$method($($args)*));
                let remaining: $ty = $remaining;
                assert_eq!(remaining, slice);
            }
        )*
    };
}

split_off_tests! {
    slice: &[0, 1, 2, 3], method: split_off,
    (split_off_in_bounds_range_to, (..1), Some(&[0] as _), &[1, 2, 3]),
    (split_off_in_bounds_range_to_inclusive, (..=0), Some(&[0] as _), &[1, 2, 3]),
    (split_off_in_bounds_range_from, (2..), Some(&[2, 3] as _), &[0, 1]),
    (split_off_oob_range_to, (..5), None, &[0, 1, 2, 3]),
    (split_off_oob_range_to_inclusive, (..=4), None, &[0, 1, 2, 3]),
    (split_off_oob_range_from, (5..), None, &[0, 1, 2, 3]),
}

split_off_tests! {
    slice: &mut [0, 1, 2, 3], method: split_off_mut,
    (split_off_mut_in_bounds_range_to, (..1), Some(&mut [0] as _), &mut [1, 2, 3]),
    (split_off_mut_in_bounds_range_to_inclusive, (..=0), Some(&mut [0] as _), &mut [1, 2, 3]),
    (split_off_mut_in_bounds_range_from, (2..), Some(&mut [2, 3] as _), &mut [0, 1]),
    (split_off_mut_oob_range_to, (..5), None, &mut [0, 1, 2, 3]),
    (split_off_mut_oob_range_to_inclusive, (..=4), None, &mut [0, 1, 2, 3]),
    (split_off_mut_oob_range_from, (5..), None, &mut [0, 1, 2, 3]),
}

split_off_tests! {
    slice: &[1, 2], method: split_off_first,
    (split_off_first_nonempty, (), Some(&1), &[2]),
}

split_off_tests! {
    slice: &mut [1, 2], method: split_off_first_mut,
    (split_off_first_mut_nonempty, (), Some(&mut 1), &mut [2]),
}

split_off_tests! {
    slice: &[1, 2], method: split_off_last,
    (split_off_last_nonempty, (), Some(&2), &[1]),
}

split_off_tests! {
    slice: &mut [1, 2], method: split_off_last_mut,
    (split_off_last_mut_nonempty, (), Some(&mut 2), &mut [1]),
}

split_off_tests! {
    slice: &[], method: split_off_first,
    (split_off_first_empty, (), None, &[]),
}

split_off_tests! {
    slice: &mut [], method: split_off_first_mut,
    (split_off_first_mut_empty, (), None, &mut []),
}

split_off_tests! {
    slice: &[], method: split_off_last,
    (split_off_last_empty, (), None, &[]),
}

split_off_tests! {
    slice: &mut [], method: split_off_last_mut,
    (split_off_last_mut_empty, (), None, &mut []),
}

#[cfg(not(miri))] // unused in Miri
const EMPTY_MAX: &'static [()] = &[(); usize::MAX];

// can't be a constant due to const mutability rules
#[cfg(not(miri))] // unused in Miri
macro_rules! empty_max_mut {
    () => {
        &mut [(); usize::MAX] as _
    };
}

#[cfg(not(miri))] // Comparing usize::MAX many elements takes forever in Miri (and in rustc without optimizations)
split_off_tests! {
    slice: &[(); usize::MAX], method: split_off,
    (split_off_in_bounds_max_range_to, (..usize::MAX), Some(EMPTY_MAX), &[(); 0]),
    (split_off_oob_max_range_to_inclusive, (..=usize::MAX), None, EMPTY_MAX),
    (split_off_in_bounds_max_range_from, (usize::MAX..), Some(&[] as _), EMPTY_MAX),
}

#[cfg(not(miri))] // Comparing usize::MAX many elements takes forever in Miri (and in rustc without optimizations)
split_off_tests! {
    slice: &mut [(); usize::MAX], method: split_off_mut,
    (split_off_mut_in_bounds_max_range_to, (..usize::MAX), Some(empty_max_mut!()), &mut [(); 0]),
    (split_off_mut_oob_max_range_to_inclusive, (..=usize::MAX), None, empty_max_mut!()),
    (split_off_mut_in_bounds_max_range_from, (usize::MAX..), Some(&mut [] as _), empty_max_mut!()),
}

#[test]
fn test_slice_from_ptr_range() {
    let arr = ["foo".to_owned(), "bar".to_owned()];
    let range = arr.as_ptr_range();
    unsafe {
        assert_eq!(slice::from_ptr_range(range), &arr);
    }

    let mut arr = [1, 2, 3];
    let range = arr.as_mut_ptr_range();
    unsafe {
        assert_eq!(slice::from_mut_ptr_range(range), &mut [1, 2, 3]);
    }

    let arr: [Vec<String>; 0] = [];
    let range = arr.as_ptr_range();
    unsafe {
        assert_eq!(slice::from_ptr_range(range), &arr);
    }
}

#[test]
#[should_panic = "slice len overflow"]
fn test_flatten_size_overflow() {
    let x = &[[(); usize::MAX]; 2][..];
    let _ = x.as_flattened();
}

#[test]
#[should_panic = "slice len overflow"]
fn test_flatten_mut_size_overflow() {
    let x = &mut [[(); usize::MAX]; 2][..];
    let _ = x.as_flattened_mut();
}

#[test]
fn test_get_disjoint_mut_normal_2() {
    let mut v = vec![1, 2, 3, 4, 5];
    let [a, b] = v.get_disjoint_mut([3, 0]).unwrap();
    *a += 10;
    *b += 100;
    assert_eq!(v, vec![101, 2, 3, 14, 5]);

    let [a, b] = v.get_disjoint_mut([0..=1, 2..=2]).unwrap();
    assert_eq!(a, &mut [101, 2][..]);
    assert_eq!(b, &mut [3][..]);
    a[0] += 10;
    a[1] += 20;
    b[0] += 100;
    assert_eq!(v, vec![111, 22, 103, 14, 5]);
}

#[test]
fn test_get_disjoint_mut_normal_3() {
    let mut v = vec![1, 2, 3, 4, 5];
    let [a, b, c] = v.get_disjoint_mut([0, 4, 2]).unwrap();
    *a += 10;
    *b += 100;
    *c += 1000;
    assert_eq!(v, vec![11, 2, 1003, 4, 105]);

    let [a, b, c] = v.get_disjoint_mut([0..1, 4..5, 1..4]).unwrap();
    assert_eq!(a, &mut [11][..]);
    assert_eq!(b, &mut [105][..]);
    assert_eq!(c, &mut [2, 1003, 4][..]);
    a[0] += 10;
    b[0] += 100;
    c[0] += 1000;
    assert_eq!(v, vec![21, 1002, 1003, 4, 205]);
}

#[test]
fn test_get_disjoint_mut_empty() {
    let mut v = vec![1, 2, 3, 4, 5];
    let [] = v.get_disjoint_mut::<usize, 0>([]).unwrap();
    let [] = v.get_disjoint_mut::<RangeInclusive<usize>, 0>([]).unwrap();
    let [] = v.get_disjoint_mut::<Range<usize>, 0>([]).unwrap();
    assert_eq!(v, vec![1, 2, 3, 4, 5]);
}

#[test]
fn test_get_disjoint_mut_single_first() {
    let mut v = vec![1, 2, 3, 4, 5];
    let [a] = v.get_disjoint_mut([0]).unwrap();
    *a += 10;
    assert_eq!(v, vec![11, 2, 3, 4, 5]);
}

#[test]
fn test_get_disjoint_mut_single_last() {
    let mut v = vec![1, 2, 3, 4, 5];
    let [a] = v.get_disjoint_mut([4]).unwrap();
    *a += 10;
    assert_eq!(v, vec![1, 2, 3, 4, 15]);
}

#[test]
fn test_get_disjoint_mut_oob_nonempty() {
    let mut v = vec![1, 2, 3, 4, 5];
    assert!(v.get_disjoint_mut([5]).is_err());
}

#[test]
fn test_get_disjoint_mut_oob_empty() {
    let mut v: Vec<i32> = vec![];
    assert!(v.get_disjoint_mut([0]).is_err());
}

#[test]
fn test_get_disjoint_mut_duplicate() {
    let mut v = vec![1, 2, 3, 4, 5];
    assert!(v.get_disjoint_mut([1, 3, 3, 4]).is_err());
}

#[test]
fn test_get_disjoint_mut_range_oob() {
    let mut v = vec![1, 2, 3, 4, 5];
    assert!(v.get_disjoint_mut([0..6]).is_err());
    assert!(v.get_disjoint_mut([5..6]).is_err());
    assert!(v.get_disjoint_mut([6..6]).is_err());
    assert!(v.get_disjoint_mut([0..=5]).is_err());
    assert!(v.get_disjoint_mut([0..=6]).is_err());
    assert!(v.get_disjoint_mut([5..=5]).is_err());
}

#[test]
fn test_get_disjoint_mut_range_overlapping() {
    let mut v = vec![1, 2, 3, 4, 5];
    assert!(v.get_disjoint_mut([0..1, 0..2]).is_err());
    assert!(v.get_disjoint_mut([0..1, 1..2, 0..1]).is_err());
    assert!(v.get_disjoint_mut([0..3, 1..1]).is_err());
    assert!(v.get_disjoint_mut([0..3, 1..2]).is_err());
    assert!(v.get_disjoint_mut([0..=0, 2..=2, 0..=1]).is_err());
    assert!(v.get_disjoint_mut([0..=4, 0..=0]).is_err());
    assert!(v.get_disjoint_mut([4..=4, 0..=0, 3..=4]).is_err());
}

#[test]
fn test_get_disjoint_mut_range_empty_at_edge() {
    let mut v = vec![1, 2, 3, 4, 5];
    assert_eq!(
        v.get_disjoint_mut([0..0, 0..5, 5..5]),
        Ok([&mut [][..], &mut [1, 2, 3, 4, 5], &mut []]),
    );
    assert_eq!(
        v.get_disjoint_mut([0..0, 0..1, 1..1, 1..2, 2..2, 2..3, 3..3, 3..4, 4..4, 4..5, 5..5]),
        Ok([
            &mut [][..],
            &mut [1],
            &mut [],
            &mut [2],
            &mut [],
            &mut [3],
            &mut [],
            &mut [4],
            &mut [],
            &mut [5],
            &mut [],
        ]),
    );
}

#[test]
fn test_slice_from_raw_parts_in_const() {
    static FANCY: i32 = 4;
    static FANCY_SLICE: &[i32] = unsafe { std::slice::from_raw_parts(&FANCY, 1) };
    assert_eq!(FANCY_SLICE.as_ptr(), std::ptr::addr_of!(FANCY));
    assert_eq!(FANCY_SLICE.len(), 1);

    const EMPTY_SLICE: &[i32] =
        unsafe { std::slice::from_raw_parts(std::ptr::without_provenance(123456), 0) };
    assert_eq!(EMPTY_SLICE.as_ptr().addr(), 123456);
    assert_eq!(EMPTY_SLICE.len(), 0);
}
