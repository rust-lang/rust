// ignore-tidy-filelength

use core::cell::Cell;
use core::convert::TryFrom;
use core::iter::*;

/// An iterator wrapper that panics whenever `next` or `next_back` is called
/// after `None` has been returned.
struct Unfuse<I> {
    iter: I,
    exhausted: bool,
}

fn unfuse<I: IntoIterator>(iter: I) -> Unfuse<I::IntoIter> {
    Unfuse { iter: iter.into_iter(), exhausted: false }
}

impl<I> Iterator for Unfuse<I>
where
    I: Iterator,
{
    type Item = I::Item;

    fn next(&mut self) -> Option<Self::Item> {
        assert!(!self.exhausted);
        let next = self.iter.next();
        self.exhausted = next.is_none();
        next
    }
}

impl<I> DoubleEndedIterator for Unfuse<I>
where
    I: DoubleEndedIterator,
{
    fn next_back(&mut self) -> Option<Self::Item> {
        assert!(!self.exhausted);
        let next = self.iter.next_back();
        self.exhausted = next.is_none();
        next
    }
}

#[test]
fn test_lt() {
    let empty: [isize; 0] = [];
    let xs = [1, 2, 3];
    let ys = [1, 2, 0];

    assert!(!xs.iter().lt(ys.iter()));
    assert!(!xs.iter().le(ys.iter()));
    assert!(xs.iter().gt(ys.iter()));
    assert!(xs.iter().ge(ys.iter()));

    assert!(ys.iter().lt(xs.iter()));
    assert!(ys.iter().le(xs.iter()));
    assert!(!ys.iter().gt(xs.iter()));
    assert!(!ys.iter().ge(xs.iter()));

    assert!(empty.iter().lt(xs.iter()));
    assert!(empty.iter().le(xs.iter()));
    assert!(!empty.iter().gt(xs.iter()));
    assert!(!empty.iter().ge(xs.iter()));

    // Sequence with NaN
    let u = [1.0f64, 2.0];
    let v = [0.0f64 / 0.0, 3.0];

    assert!(!u.iter().lt(v.iter()));
    assert!(!u.iter().le(v.iter()));
    assert!(!u.iter().gt(v.iter()));
    assert!(!u.iter().ge(v.iter()));

    let a = [0.0f64 / 0.0];
    let b = [1.0f64];
    let c = [2.0f64];

    assert!(a.iter().lt(b.iter()) == (a[0] < b[0]));
    assert!(a.iter().le(b.iter()) == (a[0] <= b[0]));
    assert!(a.iter().gt(b.iter()) == (a[0] > b[0]));
    assert!(a.iter().ge(b.iter()) == (a[0] >= b[0]));

    assert!(c.iter().lt(b.iter()) == (c[0] < b[0]));
    assert!(c.iter().le(b.iter()) == (c[0] <= b[0]));
    assert!(c.iter().gt(b.iter()) == (c[0] > b[0]));
    assert!(c.iter().ge(b.iter()) == (c[0] >= b[0]));
}

#[test]
fn test_multi_iter() {
    let xs = [1, 2, 3, 4];
    let ys = [4, 3, 2, 1];
    assert!(xs.iter().eq(ys.iter().rev()));
    assert!(xs.iter().lt(xs.iter().skip(2)));
}

#[test]
fn test_cmp_by() {
    use core::cmp::Ordering;

    let f = |x: i32, y: i32| (x * x).cmp(&y);
    let xs = || [1, 2, 3, 4].iter().copied();
    let ys = || [1, 4, 16].iter().copied();

    assert_eq!(xs().cmp_by(ys(), f), Ordering::Less);
    assert_eq!(ys().cmp_by(xs(), f), Ordering::Greater);
    assert_eq!(xs().cmp_by(xs().map(|x| x * x), f), Ordering::Equal);
    assert_eq!(xs().rev().cmp_by(ys().rev(), f), Ordering::Greater);
    assert_eq!(xs().cmp_by(ys().rev(), f), Ordering::Less);
    assert_eq!(xs().cmp_by(ys().take(2), f), Ordering::Greater);
}

#[test]
fn test_partial_cmp_by() {
    use core::cmp::Ordering;

    let f = |x: i32, y: i32| (x * x).partial_cmp(&y);
    let xs = || [1, 2, 3, 4].iter().copied();
    let ys = || [1, 4, 16].iter().copied();

    assert_eq!(xs().partial_cmp_by(ys(), f), Some(Ordering::Less));
    assert_eq!(ys().partial_cmp_by(xs(), f), Some(Ordering::Greater));
    assert_eq!(xs().partial_cmp_by(xs().map(|x| x * x), f), Some(Ordering::Equal));
    assert_eq!(xs().rev().partial_cmp_by(ys().rev(), f), Some(Ordering::Greater));
    assert_eq!(xs().partial_cmp_by(xs().rev(), f), Some(Ordering::Less));
    assert_eq!(xs().partial_cmp_by(ys().take(2), f), Some(Ordering::Greater));

    let f = |x: f64, y: f64| (x * x).partial_cmp(&y);
    let xs = || [1.0, 2.0, 3.0, 4.0].iter().copied();
    let ys = || [1.0, 4.0, f64::NAN, 16.0].iter().copied();

    assert_eq!(xs().partial_cmp_by(ys(), f), None);
    assert_eq!(ys().partial_cmp_by(xs(), f), Some(Ordering::Greater));
}

#[test]
fn test_eq_by() {
    let f = |x: i32, y: i32| x * x == y;
    let xs = || [1, 2, 3, 4].iter().copied();
    let ys = || [1, 4, 9, 16].iter().copied();

    assert!(xs().eq_by(ys(), f));
    assert!(!ys().eq_by(xs(), f));
    assert!(!xs().eq_by(xs(), f));
    assert!(!ys().eq_by(ys(), f));

    assert!(!xs().take(3).eq_by(ys(), f));
    assert!(!xs().eq_by(ys().take(3), f));
    assert!(xs().take(3).eq_by(ys().take(3), f));
}

#[test]
fn test_counter_from_iter() {
    let it = (0..).step_by(5).take(10);
    let xs: Vec<isize> = FromIterator::from_iter(it);
    assert_eq!(xs, [0, 5, 10, 15, 20, 25, 30, 35, 40, 45]);
}

#[test]
fn test_iterator_chain() {
    let xs = [0, 1, 2, 3, 4, 5];
    let ys = [30, 40, 50, 60];
    let expected = [0, 1, 2, 3, 4, 5, 30, 40, 50, 60];
    let it = xs.iter().chain(&ys);
    let mut i = 0;
    for &x in it {
        assert_eq!(x, expected[i]);
        i += 1;
    }
    assert_eq!(i, expected.len());

    let ys = (30..).step_by(10).take(4);
    let it = xs.iter().cloned().chain(ys);
    let mut i = 0;
    for x in it {
        assert_eq!(x, expected[i]);
        i += 1;
    }
    assert_eq!(i, expected.len());
}

#[test]
fn test_iterator_chain_advance_by() {
    fn test_chain(xs: &[i32], ys: &[i32]) {
        let len = xs.len() + ys.len();

        for i in 0..xs.len() {
            let mut iter = unfuse(xs).chain(unfuse(ys));
            iter.advance_by(i).unwrap();
            assert_eq!(iter.next(), Some(&xs[i]));
            assert_eq!(iter.advance_by(100), Err(len - i - 1));
        }

        for i in 0..ys.len() {
            let mut iter = unfuse(xs).chain(unfuse(ys));
            iter.advance_by(xs.len() + i).unwrap();
            assert_eq!(iter.next(), Some(&ys[i]));
            assert_eq!(iter.advance_by(100), Err(ys.len() - i - 1));
        }

        let mut iter = xs.iter().chain(ys);
        iter.advance_by(len).unwrap();
        assert_eq!(iter.next(), None);

        let mut iter = xs.iter().chain(ys);
        assert_eq!(iter.advance_by(len + 1), Err(len));
    }

    test_chain(&[], &[]);
    test_chain(&[], &[0, 1, 2, 3, 4, 5]);
    test_chain(&[0, 1, 2, 3, 4, 5], &[]);
    test_chain(&[0, 1, 2, 3, 4, 5], &[30, 40, 50, 60]);
}

#[test]
fn test_iterator_chain_advance_back_by() {
    fn test_chain(xs: &[i32], ys: &[i32]) {
        let len = xs.len() + ys.len();

        for i in 0..ys.len() {
            let mut iter = unfuse(xs).chain(unfuse(ys));
            iter.advance_back_by(i).unwrap();
            assert_eq!(iter.next_back(), Some(&ys[ys.len() - i - 1]));
            assert_eq!(iter.advance_back_by(100), Err(len - i - 1));
        }

        for i in 0..xs.len() {
            let mut iter = unfuse(xs).chain(unfuse(ys));
            iter.advance_back_by(ys.len() + i).unwrap();
            assert_eq!(iter.next_back(), Some(&xs[xs.len() - i - 1]));
            assert_eq!(iter.advance_back_by(100), Err(xs.len() - i - 1));
        }

        let mut iter = xs.iter().chain(ys);
        iter.advance_back_by(len).unwrap();
        assert_eq!(iter.next_back(), None);

        let mut iter = xs.iter().chain(ys);
        assert_eq!(iter.advance_back_by(len + 1), Err(len));
    }

    test_chain(&[], &[]);
    test_chain(&[], &[0, 1, 2, 3, 4, 5]);
    test_chain(&[0, 1, 2, 3, 4, 5], &[]);
    test_chain(&[0, 1, 2, 3, 4, 5], &[30, 40, 50, 60]);
}

#[test]
fn test_iterator_chain_nth() {
    let xs = [0, 1, 2, 3, 4, 5];
    let ys = [30, 40, 50, 60];
    let zs = [];
    let expected = [0, 1, 2, 3, 4, 5, 30, 40, 50, 60];
    for (i, x) in expected.iter().enumerate() {
        assert_eq!(Some(x), xs.iter().chain(&ys).nth(i));
    }
    assert_eq!(zs.iter().chain(&xs).nth(0), Some(&0));

    let mut it = xs.iter().chain(&zs);
    assert_eq!(it.nth(5), Some(&5));
    assert_eq!(it.next(), None);
}

#[test]
fn test_iterator_chain_nth_back() {
    let xs = [0, 1, 2, 3, 4, 5];
    let ys = [30, 40, 50, 60];
    let zs = [];
    let expected = [0, 1, 2, 3, 4, 5, 30, 40, 50, 60];
    for (i, x) in expected.iter().rev().enumerate() {
        assert_eq!(Some(x), xs.iter().chain(&ys).nth_back(i));
    }
    assert_eq!(zs.iter().chain(&xs).nth_back(0), Some(&5));

    let mut it = xs.iter().chain(&zs);
    assert_eq!(it.nth_back(5), Some(&0));
    assert_eq!(it.next(), None);
}

#[test]
fn test_iterator_chain_last() {
    let xs = [0, 1, 2, 3, 4, 5];
    let ys = [30, 40, 50, 60];
    let zs = [];
    assert_eq!(xs.iter().chain(&ys).last(), Some(&60));
    assert_eq!(zs.iter().chain(&ys).last(), Some(&60));
    assert_eq!(ys.iter().chain(&zs).last(), Some(&60));
    assert_eq!(zs.iter().chain(&zs).last(), None);
}

#[test]
fn test_iterator_chain_count() {
    let xs = [0, 1, 2, 3, 4, 5];
    let ys = [30, 40, 50, 60];
    let zs = [];
    assert_eq!(xs.iter().chain(&ys).count(), 10);
    assert_eq!(zs.iter().chain(&ys).count(), 4);
}

#[test]
fn test_iterator_chain_find() {
    let xs = [0, 1, 2, 3, 4, 5];
    let ys = [30, 40, 50, 60];
    let mut iter = xs.iter().chain(&ys);
    assert_eq!(iter.find(|&&i| i == 4), Some(&4));
    assert_eq!(iter.next(), Some(&5));
    assert_eq!(iter.find(|&&i| i == 40), Some(&40));
    assert_eq!(iter.next(), Some(&50));
    assert_eq!(iter.find(|&&i| i == 100), None);
    assert_eq!(iter.next(), None);
}

struct Toggle {
    is_empty: bool,
}

impl Iterator for Toggle {
    type Item = ();

    // alternates between `None` and `Some(())`
    fn next(&mut self) -> Option<Self::Item> {
        if self.is_empty {
            self.is_empty = false;
            None
        } else {
            self.is_empty = true;
            Some(())
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        if self.is_empty { (0, Some(0)) } else { (1, Some(1)) }
    }
}

impl DoubleEndedIterator for Toggle {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.next()
    }
}

#[test]
fn test_iterator_chain_size_hint() {
    // this chains an iterator of length 0 with an iterator of length 1,
    // so after calling `.next()` once, the iterator is empty and the
    // state is `ChainState::Back`. `.size_hint()` should now disregard
    // the size hint of the left iterator
    let mut iter = Toggle { is_empty: true }.chain(once(()));
    assert_eq!(iter.next(), Some(()));
    assert_eq!(iter.size_hint(), (0, Some(0)));

    let mut iter = once(()).chain(Toggle { is_empty: true });
    assert_eq!(iter.next_back(), Some(()));
    assert_eq!(iter.size_hint(), (0, Some(0)));
}

#[test]
fn test_iterator_chain_unfused() {
    // Chain shouldn't be fused in its second iterator, depending on direction
    let mut iter = NonFused::new(empty()).chain(Toggle { is_empty: true });
    iter.next().unwrap_none();
    iter.next().unwrap();
    iter.next().unwrap_none();

    let mut iter = Toggle { is_empty: true }.chain(NonFused::new(empty()));
    iter.next_back().unwrap_none();
    iter.next_back().unwrap();
    iter.next_back().unwrap_none();
}

#[test]
fn test_zip_nth() {
    let xs = [0, 1, 2, 4, 5];
    let ys = [10, 11, 12];

    let mut it = xs.iter().zip(&ys);
    assert_eq!(it.nth(0), Some((&0, &10)));
    assert_eq!(it.nth(1), Some((&2, &12)));
    assert_eq!(it.nth(0), None);

    let mut it = xs.iter().zip(&ys);
    assert_eq!(it.nth(3), None);

    let mut it = ys.iter().zip(&xs);
    assert_eq!(it.nth(3), None);
}

#[test]
fn test_zip_nth_side_effects() {
    let mut a = Vec::new();
    let mut b = Vec::new();
    let value = [1, 2, 3, 4, 5, 6]
        .iter()
        .cloned()
        .map(|n| {
            a.push(n);
            n * 10
        })
        .zip([2, 3, 4, 5, 6, 7, 8].iter().cloned().map(|n| {
            b.push(n * 100);
            n * 1000
        }))
        .skip(1)
        .nth(3);
    assert_eq!(value, Some((50, 6000)));
    assert_eq!(a, vec![1, 2, 3, 4, 5]);
    assert_eq!(b, vec![200, 300, 400, 500, 600]);
}

#[test]
fn test_zip_next_back_side_effects() {
    let mut a = Vec::new();
    let mut b = Vec::new();
    let mut iter = [1, 2, 3, 4, 5, 6]
        .iter()
        .cloned()
        .map(|n| {
            a.push(n);
            n * 10
        })
        .zip([2, 3, 4, 5, 6, 7, 8].iter().cloned().map(|n| {
            b.push(n * 100);
            n * 1000
        }));

    // The second iterator is one item longer, so `next_back` is called on it
    // one more time.
    assert_eq!(iter.next_back(), Some((60, 7000)));
    assert_eq!(iter.next_back(), Some((50, 6000)));
    assert_eq!(iter.next_back(), Some((40, 5000)));
    assert_eq!(iter.next_back(), Some((30, 4000)));
    assert_eq!(a, vec![6, 5, 4, 3]);
    assert_eq!(b, vec![800, 700, 600, 500, 400]);
}

#[test]
fn test_zip_nth_back_side_effects() {
    let mut a = Vec::new();
    let mut b = Vec::new();
    let value = [1, 2, 3, 4, 5, 6]
        .iter()
        .cloned()
        .map(|n| {
            a.push(n);
            n * 10
        })
        .zip([2, 3, 4, 5, 6, 7, 8].iter().cloned().map(|n| {
            b.push(n * 100);
            n * 1000
        }))
        .nth_back(3);
    assert_eq!(value, Some((30, 4000)));
    assert_eq!(a, vec![6, 5, 4, 3]);
    assert_eq!(b, vec![800, 700, 600, 500, 400]);
}

#[test]
fn test_zip_next_back_side_effects_exhausted() {
    let mut a = Vec::new();
    let mut b = Vec::new();
    let mut iter = [1, 2, 3, 4, 5, 6]
        .iter()
        .cloned()
        .map(|n| {
            a.push(n);
            n * 10
        })
        .zip([2, 3, 4].iter().cloned().map(|n| {
            b.push(n * 100);
            n * 1000
        }));

    iter.next();
    iter.next();
    iter.next();
    iter.next();
    assert_eq!(iter.next_back(), None);
    assert_eq!(a, vec![1, 2, 3, 4, 6, 5]);
    assert_eq!(b, vec![200, 300, 400]);
}

#[derive(Debug)]
struct CountClone(Cell<i32>);

fn count_clone() -> CountClone {
    CountClone(Cell::new(0))
}

impl PartialEq<i32> for CountClone {
    fn eq(&self, rhs: &i32) -> bool {
        self.0.get() == *rhs
    }
}

impl Clone for CountClone {
    fn clone(&self) -> Self {
        let ret = CountClone(self.0.clone());
        let n = self.0.get();
        self.0.set(n + 1);
        ret
    }
}

#[test]
fn test_zip_cloned_sideffectful() {
    let xs = [count_clone(), count_clone(), count_clone(), count_clone()];
    let ys = [count_clone(), count_clone()];

    for _ in xs.iter().cloned().zip(ys.iter().cloned()) {}

    assert_eq!(&xs, &[1, 1, 1, 0][..]);
    assert_eq!(&ys, &[1, 1][..]);

    let xs = [count_clone(), count_clone()];
    let ys = [count_clone(), count_clone(), count_clone(), count_clone()];

    for _ in xs.iter().cloned().zip(ys.iter().cloned()) {}

    assert_eq!(&xs, &[1, 1][..]);
    assert_eq!(&ys, &[1, 1, 0, 0][..]);
}

#[test]
fn test_zip_map_sideffectful() {
    let mut xs = [0; 6];
    let mut ys = [0; 4];

    for _ in xs.iter_mut().map(|x| *x += 1).zip(ys.iter_mut().map(|y| *y += 1)) {}

    assert_eq!(&xs, &[1, 1, 1, 1, 1, 0]);
    assert_eq!(&ys, &[1, 1, 1, 1]);

    let mut xs = [0; 4];
    let mut ys = [0; 6];

    for _ in xs.iter_mut().map(|x| *x += 1).zip(ys.iter_mut().map(|y| *y += 1)) {}

    assert_eq!(&xs, &[1, 1, 1, 1]);
    assert_eq!(&ys, &[1, 1, 1, 1, 0, 0]);
}

#[test]
fn test_zip_map_rev_sideffectful() {
    let mut xs = [0; 6];
    let mut ys = [0; 4];

    {
        let mut it = xs.iter_mut().map(|x| *x += 1).zip(ys.iter_mut().map(|y| *y += 1));
        it.next_back();
    }
    assert_eq!(&xs, &[0, 0, 0, 1, 1, 1]);
    assert_eq!(&ys, &[0, 0, 0, 1]);

    let mut xs = [0; 6];
    let mut ys = [0; 4];

    {
        let mut it = xs.iter_mut().map(|x| *x += 1).zip(ys.iter_mut().map(|y| *y += 1));
        (&mut it).take(5).count();
        it.next_back();
    }
    assert_eq!(&xs, &[1, 1, 1, 1, 1, 1]);
    assert_eq!(&ys, &[1, 1, 1, 1]);
}

#[test]
fn test_zip_nested_sideffectful() {
    let mut xs = [0; 6];
    let ys = [0; 4];

    {
        // test that it has the side effect nested inside enumerate
        let it = xs.iter_mut().map(|x| *x = 1).enumerate().zip(&ys);
        it.count();
    }
    assert_eq!(&xs, &[1, 1, 1, 1, 1, 0]);
}

#[test]
fn test_zip_nth_back_side_effects_exhausted() {
    let mut a = Vec::new();
    let mut b = Vec::new();
    let mut iter = [1, 2, 3, 4, 5, 6]
        .iter()
        .cloned()
        .map(|n| {
            a.push(n);
            n * 10
        })
        .zip([2, 3, 4].iter().cloned().map(|n| {
            b.push(n * 100);
            n * 1000
        }));

    iter.next();
    iter.next();
    iter.next();
    iter.next();
    assert_eq!(iter.nth_back(0), None);
    assert_eq!(a, vec![1, 2, 3, 4, 6, 5]);
    assert_eq!(b, vec![200, 300, 400]);
}

#[test]
fn test_iterator_step_by() {
    // Identity
    let mut it = (0..).step_by(1).take(3);
    assert_eq!(it.next(), Some(0));
    assert_eq!(it.next(), Some(1));
    assert_eq!(it.next(), Some(2));
    assert_eq!(it.next(), None);

    let mut it = (0..).step_by(3).take(4);
    assert_eq!(it.next(), Some(0));
    assert_eq!(it.next(), Some(3));
    assert_eq!(it.next(), Some(6));
    assert_eq!(it.next(), Some(9));
    assert_eq!(it.next(), None);

    let mut it = (0..3).step_by(1);
    assert_eq!(it.next_back(), Some(2));
    assert_eq!(it.next_back(), Some(1));
    assert_eq!(it.next_back(), Some(0));
    assert_eq!(it.next_back(), None);

    let mut it = (0..11).step_by(3);
    assert_eq!(it.next_back(), Some(9));
    assert_eq!(it.next_back(), Some(6));
    assert_eq!(it.next_back(), Some(3));
    assert_eq!(it.next_back(), Some(0));
    assert_eq!(it.next_back(), None);
}

#[test]
fn test_iterator_step_by_nth() {
    let mut it = (0..16).step_by(5);
    assert_eq!(it.nth(0), Some(0));
    assert_eq!(it.nth(0), Some(5));
    assert_eq!(it.nth(0), Some(10));
    assert_eq!(it.nth(0), Some(15));
    assert_eq!(it.nth(0), None);

    let it = (0..18).step_by(5);
    assert_eq!(it.clone().nth(0), Some(0));
    assert_eq!(it.clone().nth(1), Some(5));
    assert_eq!(it.clone().nth(2), Some(10));
    assert_eq!(it.clone().nth(3), Some(15));
    assert_eq!(it.clone().nth(4), None);
    assert_eq!(it.clone().nth(42), None);
}

#[test]
fn test_iterator_step_by_nth_overflow() {
    #[cfg(target_pointer_width = "8")]
    type Bigger = u16;
    #[cfg(target_pointer_width = "16")]
    type Bigger = u32;
    #[cfg(target_pointer_width = "32")]
    type Bigger = u64;
    #[cfg(target_pointer_width = "64")]
    type Bigger = u128;

    #[derive(Clone)]
    struct Test(Bigger);
    impl Iterator for &mut Test {
        type Item = i32;
        fn next(&mut self) -> Option<Self::Item> {
            Some(21)
        }
        fn nth(&mut self, n: usize) -> Option<Self::Item> {
            self.0 += n as Bigger + 1;
            Some(42)
        }
    }

    let mut it = Test(0);
    let root = usize::MAX >> (usize::BITS / 2);
    let n = root + 20;
    (&mut it).step_by(n).nth(n);
    assert_eq!(it.0, n as Bigger * n as Bigger);

    // large step
    let mut it = Test(0);
    (&mut it).step_by(usize::MAX).nth(5);
    assert_eq!(it.0, (usize::MAX as Bigger) * 5);

    // n + 1 overflows
    let mut it = Test(0);
    (&mut it).step_by(2).nth(usize::MAX);
    assert_eq!(it.0, (usize::MAX as Bigger) * 2);

    // n + 1 overflows
    let mut it = Test(0);
    (&mut it).step_by(1).nth(usize::MAX);
    assert_eq!(it.0, (usize::MAX as Bigger) * 1);
}

#[test]
fn test_iterator_step_by_nth_try_fold() {
    let mut it = (0..).step_by(10);
    assert_eq!(it.try_fold(0, i8::checked_add), None);
    assert_eq!(it.next(), Some(60));
    assert_eq!(it.try_fold(0, i8::checked_add), None);
    assert_eq!(it.next(), Some(90));

    let mut it = (100..).step_by(10);
    assert_eq!(it.try_fold(50, i8::checked_add), None);
    assert_eq!(it.next(), Some(110));

    let mut it = (100..=100).step_by(10);
    assert_eq!(it.next(), Some(100));
    assert_eq!(it.try_fold(0, i8::checked_add), Some(0));
}

#[test]
fn test_iterator_step_by_nth_back() {
    let mut it = (0..16).step_by(5);
    assert_eq!(it.nth_back(0), Some(15));
    assert_eq!(it.nth_back(0), Some(10));
    assert_eq!(it.nth_back(0), Some(5));
    assert_eq!(it.nth_back(0), Some(0));
    assert_eq!(it.nth_back(0), None);

    let mut it = (0..16).step_by(5);
    assert_eq!(it.next(), Some(0)); // to set `first_take` to `false`
    assert_eq!(it.nth_back(0), Some(15));
    assert_eq!(it.nth_back(0), Some(10));
    assert_eq!(it.nth_back(0), Some(5));
    assert_eq!(it.nth_back(0), None);

    let it = || (0..18).step_by(5);
    assert_eq!(it().nth_back(0), Some(15));
    assert_eq!(it().nth_back(1), Some(10));
    assert_eq!(it().nth_back(2), Some(5));
    assert_eq!(it().nth_back(3), Some(0));
    assert_eq!(it().nth_back(4), None);
    assert_eq!(it().nth_back(42), None);
}

#[test]
fn test_iterator_step_by_nth_try_rfold() {
    let mut it = (0..100).step_by(10);
    assert_eq!(it.try_rfold(0, i8::checked_add), None);
    assert_eq!(it.next_back(), Some(70));
    assert_eq!(it.next(), Some(0));
    assert_eq!(it.try_rfold(0, i8::checked_add), None);
    assert_eq!(it.next_back(), Some(30));

    let mut it = (0..100).step_by(10);
    assert_eq!(it.try_rfold(50, i8::checked_add), None);
    assert_eq!(it.next_back(), Some(80));

    let mut it = (100..=100).step_by(10);
    assert_eq!(it.next_back(), Some(100));
    assert_eq!(it.try_fold(0, i8::checked_add), Some(0));
}

#[test]
#[should_panic]
fn test_iterator_step_by_zero() {
    let mut it = (0..).step_by(0);
    it.next();
}

#[test]
fn test_iterator_step_by_size_hint() {
    struct StubSizeHint(usize, Option<usize>);
    impl Iterator for StubSizeHint {
        type Item = ();
        fn next(&mut self) -> Option<()> {
            self.0 -= 1;
            if let Some(ref mut upper) = self.1 {
                *upper -= 1;
            }
            Some(())
        }
        fn size_hint(&self) -> (usize, Option<usize>) {
            (self.0, self.1)
        }
    }

    // The two checks in each case are needed because the logic
    // is different before the first call to `next()`.

    let mut it = StubSizeHint(10, Some(10)).step_by(1);
    assert_eq!(it.size_hint(), (10, Some(10)));
    it.next();
    assert_eq!(it.size_hint(), (9, Some(9)));

    // exact multiple
    let mut it = StubSizeHint(10, Some(10)).step_by(3);
    assert_eq!(it.size_hint(), (4, Some(4)));
    it.next();
    assert_eq!(it.size_hint(), (3, Some(3)));

    // larger base range, but not enough to get another element
    let mut it = StubSizeHint(12, Some(12)).step_by(3);
    assert_eq!(it.size_hint(), (4, Some(4)));
    it.next();
    assert_eq!(it.size_hint(), (3, Some(3)));

    // smaller base range, so fewer resulting elements
    let mut it = StubSizeHint(9, Some(9)).step_by(3);
    assert_eq!(it.size_hint(), (3, Some(3)));
    it.next();
    assert_eq!(it.size_hint(), (2, Some(2)));

    // infinite upper bound
    let mut it = StubSizeHint(usize::MAX, None).step_by(1);
    assert_eq!(it.size_hint(), (usize::MAX, None));
    it.next();
    assert_eq!(it.size_hint(), (usize::MAX - 1, None));

    // still infinite with larger step
    let mut it = StubSizeHint(7, None).step_by(3);
    assert_eq!(it.size_hint(), (3, None));
    it.next();
    assert_eq!(it.size_hint(), (2, None));

    // propagates ExactSizeIterator
    let a = [1, 2, 3, 4, 5];
    let it = a.iter().step_by(2);
    assert_eq!(it.len(), 3);

    // Cannot be TrustedLen as a step greater than one makes an iterator
    // with (usize::MAX, None) no longer meet the safety requirements
    trait TrustedLenCheck {
        fn test(self) -> bool;
    }
    impl<T: Iterator> TrustedLenCheck for T {
        default fn test(self) -> bool {
            false
        }
    }
    impl<T: TrustedLen> TrustedLenCheck for T {
        fn test(self) -> bool {
            true
        }
    }
    assert!(TrustedLenCheck::test(a.iter()));
    assert!(!TrustedLenCheck::test(a.iter().step_by(1)));
}

#[test]
fn test_filter_map() {
    let it = (0..).step_by(1).take(10).filter_map(|x| if x % 2 == 0 { Some(x * x) } else { None });
    assert_eq!(it.collect::<Vec<usize>>(), [0 * 0, 2 * 2, 4 * 4, 6 * 6, 8 * 8]);
}

#[test]
fn test_filter_map_fold() {
    let xs = [0, 1, 2, 3, 4, 5, 6, 7, 8];
    let ys = [0 * 0, 2 * 2, 4 * 4, 6 * 6, 8 * 8];
    let it = xs.iter().filter_map(|&x| if x % 2 == 0 { Some(x * x) } else { None });
    let i = it.fold(0, |i, x| {
        assert_eq!(x, ys[i]);
        i + 1
    });
    assert_eq!(i, ys.len());

    let it = xs.iter().filter_map(|&x| if x % 2 == 0 { Some(x * x) } else { None });
    let i = it.rfold(ys.len(), |i, x| {
        assert_eq!(x, ys[i - 1]);
        i - 1
    });
    assert_eq!(i, 0);
}

#[test]
fn test_iterator_enumerate() {
    let xs = [0, 1, 2, 3, 4, 5];
    let it = xs.iter().enumerate();
    for (i, &x) in it {
        assert_eq!(i, x);
    }
}

#[test]
fn test_iterator_enumerate_nth() {
    let xs = [0, 1, 2, 3, 4, 5];
    for (i, &x) in xs.iter().enumerate() {
        assert_eq!(i, x);
    }

    let mut it = xs.iter().enumerate();
    while let Some((i, &x)) = it.nth(0) {
        assert_eq!(i, x);
    }

    let mut it = xs.iter().enumerate();
    while let Some((i, &x)) = it.nth(1) {
        assert_eq!(i, x);
    }

    let (i, &x) = xs.iter().enumerate().nth(3).unwrap();
    assert_eq!(i, x);
    assert_eq!(i, 3);
}

#[test]
fn test_iterator_enumerate_nth_back() {
    let xs = [0, 1, 2, 3, 4, 5];
    let mut it = xs.iter().enumerate();
    while let Some((i, &x)) = it.nth_back(0) {
        assert_eq!(i, x);
    }

    let mut it = xs.iter().enumerate();
    while let Some((i, &x)) = it.nth_back(1) {
        assert_eq!(i, x);
    }

    let (i, &x) = xs.iter().enumerate().nth_back(3).unwrap();
    assert_eq!(i, x);
    assert_eq!(i, 2);
}

#[test]
fn test_iterator_enumerate_count() {
    let xs = [0, 1, 2, 3, 4, 5];
    assert_eq!(xs.iter().enumerate().count(), 6);
}

#[test]
fn test_iterator_enumerate_fold() {
    let xs = [0, 1, 2, 3, 4, 5];
    let mut it = xs.iter().enumerate();
    // steal a couple to get an interesting offset
    assert_eq!(it.next(), Some((0, &0)));
    assert_eq!(it.next(), Some((1, &1)));
    let i = it.fold(2, |i, (j, &x)| {
        assert_eq!(i, j);
        assert_eq!(x, xs[j]);
        i + 1
    });
    assert_eq!(i, xs.len());

    let mut it = xs.iter().enumerate();
    assert_eq!(it.next(), Some((0, &0)));
    let i = it.rfold(xs.len() - 1, |i, (j, &x)| {
        assert_eq!(i, j);
        assert_eq!(x, xs[j]);
        i - 1
    });
    assert_eq!(i, 0);
}

#[test]
fn test_iterator_filter_count() {
    let xs = [0, 1, 2, 3, 4, 5, 6, 7, 8];
    assert_eq!(xs.iter().filter(|&&x| x % 2 == 0).count(), 5);
}

#[test]
fn test_iterator_filter_fold() {
    let xs = [0, 1, 2, 3, 4, 5, 6, 7, 8];
    let ys = [0, 2, 4, 6, 8];
    let it = xs.iter().filter(|&&x| x % 2 == 0);
    let i = it.fold(0, |i, &x| {
        assert_eq!(x, ys[i]);
        i + 1
    });
    assert_eq!(i, ys.len());

    let it = xs.iter().filter(|&&x| x % 2 == 0);
    let i = it.rfold(ys.len(), |i, &x| {
        assert_eq!(x, ys[i - 1]);
        i - 1
    });
    assert_eq!(i, 0);
}

#[test]
fn test_iterator_peekable() {
    let xs = vec![0, 1, 2, 3, 4, 5];

    let mut it = xs.iter().cloned().peekable();
    assert_eq!(it.len(), 6);
    assert_eq!(it.peek().unwrap(), &0);
    assert_eq!(it.len(), 6);
    assert_eq!(it.next().unwrap(), 0);
    assert_eq!(it.len(), 5);
    assert_eq!(it.next().unwrap(), 1);
    assert_eq!(it.len(), 4);
    assert_eq!(it.next().unwrap(), 2);
    assert_eq!(it.len(), 3);
    assert_eq!(it.peek().unwrap(), &3);
    assert_eq!(it.len(), 3);
    assert_eq!(it.peek().unwrap(), &3);
    assert_eq!(it.len(), 3);
    assert_eq!(it.next().unwrap(), 3);
    assert_eq!(it.len(), 2);
    assert_eq!(it.next().unwrap(), 4);
    assert_eq!(it.len(), 1);
    assert_eq!(it.peek().unwrap(), &5);
    assert_eq!(it.len(), 1);
    assert_eq!(it.next().unwrap(), 5);
    assert_eq!(it.len(), 0);
    assert!(it.peek().is_none());
    assert_eq!(it.len(), 0);
    assert!(it.next().is_none());
    assert_eq!(it.len(), 0);

    let mut it = xs.iter().cloned().peekable();
    assert_eq!(it.len(), 6);
    assert_eq!(it.peek().unwrap(), &0);
    assert_eq!(it.len(), 6);
    assert_eq!(it.next_back().unwrap(), 5);
    assert_eq!(it.len(), 5);
    assert_eq!(it.next_back().unwrap(), 4);
    assert_eq!(it.len(), 4);
    assert_eq!(it.next_back().unwrap(), 3);
    assert_eq!(it.len(), 3);
    assert_eq!(it.peek().unwrap(), &0);
    assert_eq!(it.len(), 3);
    assert_eq!(it.peek().unwrap(), &0);
    assert_eq!(it.len(), 3);
    assert_eq!(it.next_back().unwrap(), 2);
    assert_eq!(it.len(), 2);
    assert_eq!(it.next_back().unwrap(), 1);
    assert_eq!(it.len(), 1);
    assert_eq!(it.peek().unwrap(), &0);
    assert_eq!(it.len(), 1);
    assert_eq!(it.next_back().unwrap(), 0);
    assert_eq!(it.len(), 0);
    assert!(it.peek().is_none());
    assert_eq!(it.len(), 0);
    assert!(it.next_back().is_none());
    assert_eq!(it.len(), 0);
}

#[test]
fn test_iterator_peekable_count() {
    let xs = [0, 1, 2, 3, 4, 5];
    let ys = [10];
    let zs: [i32; 0] = [];

    assert_eq!(xs.iter().peekable().count(), 6);

    let mut it = xs.iter().peekable();
    assert_eq!(it.peek(), Some(&&0));
    assert_eq!(it.count(), 6);

    assert_eq!(ys.iter().peekable().count(), 1);

    let mut it = ys.iter().peekable();
    assert_eq!(it.peek(), Some(&&10));
    assert_eq!(it.count(), 1);

    assert_eq!(zs.iter().peekable().count(), 0);

    let mut it = zs.iter().peekable();
    assert_eq!(it.peek(), None);
}

#[test]
fn test_iterator_peekable_nth() {
    let xs = [0, 1, 2, 3, 4, 5];
    let mut it = xs.iter().peekable();

    assert_eq!(it.peek(), Some(&&0));
    assert_eq!(it.nth(0), Some(&0));
    assert_eq!(it.peek(), Some(&&1));
    assert_eq!(it.nth(1), Some(&2));
    assert_eq!(it.peek(), Some(&&3));
    assert_eq!(it.nth(2), Some(&5));
    assert_eq!(it.next(), None);
}

#[test]
fn test_iterator_peekable_last() {
    let xs = [0, 1, 2, 3, 4, 5];
    let ys = [0];

    let mut it = xs.iter().peekable();
    assert_eq!(it.peek(), Some(&&0));
    assert_eq!(it.last(), Some(&5));

    let mut it = ys.iter().peekable();
    assert_eq!(it.peek(), Some(&&0));
    assert_eq!(it.last(), Some(&0));

    let mut it = ys.iter().peekable();
    assert_eq!(it.next(), Some(&0));
    assert_eq!(it.peek(), None);
    assert_eq!(it.last(), None);
}

#[test]
fn test_iterator_peekable_fold() {
    let xs = [0, 1, 2, 3, 4, 5];
    let mut it = xs.iter().peekable();
    assert_eq!(it.peek(), Some(&&0));
    let i = it.fold(0, |i, &x| {
        assert_eq!(x, xs[i]);
        i + 1
    });
    assert_eq!(i, xs.len());
}

#[test]
fn test_iterator_peekable_rfold() {
    let xs = [0, 1, 2, 3, 4, 5];
    let mut it = xs.iter().peekable();
    assert_eq!(it.peek(), Some(&&0));
    let i = it.rfold(0, |i, &x| {
        assert_eq!(x, xs[xs.len() - 1 - i]);
        i + 1
    });
    assert_eq!(i, xs.len());
}

#[test]
fn test_iterator_peekable_next_if_eq() {
    // first, try on references
    let xs = vec!["Heart", "of", "Gold"];
    let mut it = xs.into_iter().peekable();
    // try before `peek()`
    assert_eq!(it.next_if_eq(&"trillian"), None);
    assert_eq!(it.next_if_eq(&"Heart"), Some("Heart"));
    // try after peek()
    assert_eq!(it.peek(), Some(&"of"));
    assert_eq!(it.next_if_eq(&"of"), Some("of"));
    assert_eq!(it.next_if_eq(&"zaphod"), None);
    // make sure `next()` still behaves
    assert_eq!(it.next(), Some("Gold"));

    // make sure comparison works for owned values
    let xs = vec![String::from("Ludicrous"), "speed".into()];
    let mut it = xs.into_iter().peekable();
    // make sure basic functionality works
    assert_eq!(it.next_if_eq("Ludicrous"), Some("Ludicrous".into()));
    assert_eq!(it.next_if_eq("speed"), Some("speed".into()));
    assert_eq!(it.next_if_eq(""), None);
}

#[test]
fn test_iterator_peekable_mut() {
    let mut it = vec![1, 2, 3].into_iter().peekable();
    if let Some(p) = it.peek_mut() {
        if *p == 1 {
            *p = 5;
        }
    }
    assert_eq!(it.collect::<Vec<_>>(), vec![5, 2, 3]);
}

/// This is an iterator that follows the Iterator contract,
/// but it is not fused. After having returned None once, it will start
/// producing elements if .next() is called again.
pub struct CycleIter<'a, T> {
    index: usize,
    data: &'a [T],
}

pub fn cycle<T>(data: &[T]) -> CycleIter<'_, T> {
    CycleIter { index: 0, data }
}

impl<'a, T> Iterator for CycleIter<'a, T> {
    type Item = &'a T;
    fn next(&mut self) -> Option<Self::Item> {
        let elt = self.data.get(self.index);
        self.index += 1;
        self.index %= 1 + self.data.len();
        elt
    }
}

#[test]
fn test_iterator_peekable_remember_peek_none_1() {
    // Check that the loop using .peek() terminates
    let data = [1, 2, 3];
    let mut iter = cycle(&data).peekable();

    let mut n = 0;
    while let Some(_) = iter.next() {
        let is_the_last = iter.peek().is_none();
        assert_eq!(is_the_last, n == data.len() - 1);
        n += 1;
        if n > data.len() {
            break;
        }
    }
    assert_eq!(n, data.len());
}

#[test]
fn test_iterator_peekable_remember_peek_none_2() {
    let data = [0];
    let mut iter = cycle(&data).peekable();
    iter.next();
    assert_eq!(iter.peek(), None);
    assert_eq!(iter.last(), None);
}

#[test]
fn test_iterator_peekable_remember_peek_none_3() {
    let data = [0];
    let mut iter = cycle(&data).peekable();
    iter.peek();
    assert_eq!(iter.nth(0), Some(&0));

    let mut iter = cycle(&data).peekable();
    iter.next();
    assert_eq!(iter.peek(), None);
    assert_eq!(iter.nth(0), None);
}

#[test]
fn test_iterator_take_while() {
    let xs = [0, 1, 2, 3, 5, 13, 15, 16, 17, 19];
    let ys = [0, 1, 2, 3, 5, 13];
    let it = xs.iter().take_while(|&x| *x < 15);
    let mut i = 0;
    for x in it {
        assert_eq!(*x, ys[i]);
        i += 1;
    }
    assert_eq!(i, ys.len());
}

#[test]
fn test_iterator_skip_while() {
    let xs = [0, 1, 2, 3, 5, 13, 15, 16, 17, 19];
    let ys = [15, 16, 17, 19];
    let it = xs.iter().skip_while(|&x| *x < 15);
    let mut i = 0;
    for x in it {
        assert_eq!(*x, ys[i]);
        i += 1;
    }
    assert_eq!(i, ys.len());
}

#[test]
fn test_iterator_skip_while_fold() {
    let xs = [0, 1, 2, 3, 5, 13, 15, 16, 17, 19];
    let ys = [15, 16, 17, 19];
    let it = xs.iter().skip_while(|&x| *x < 15);
    let i = it.fold(0, |i, &x| {
        assert_eq!(x, ys[i]);
        i + 1
    });
    assert_eq!(i, ys.len());

    let mut it = xs.iter().skip_while(|&x| *x < 15);
    assert_eq!(it.next(), Some(&ys[0])); // process skips before folding
    let i = it.fold(1, |i, &x| {
        assert_eq!(x, ys[i]);
        i + 1
    });
    assert_eq!(i, ys.len());
}

#[test]
fn test_iterator_skip() {
    let xs = [0, 1, 2, 3, 5, 13, 15, 16, 17, 19, 20, 30];
    let ys = [13, 15, 16, 17, 19, 20, 30];
    let mut it = xs.iter().skip(5);
    let mut i = 0;
    while let Some(&x) = it.next() {
        assert_eq!(x, ys[i]);
        i += 1;
        assert_eq!(it.len(), xs.len() - 5 - i);
    }
    assert_eq!(i, ys.len());
    assert_eq!(it.len(), 0);
}

#[test]
fn test_iterator_skip_doubleended() {
    let xs = [0, 1, 2, 3, 5, 13, 15, 16, 17, 19, 20, 30];
    let mut it = xs.iter().rev().skip(5);
    assert_eq!(it.next(), Some(&15));
    assert_eq!(it.by_ref().rev().next(), Some(&0));
    assert_eq!(it.next(), Some(&13));
    assert_eq!(it.by_ref().rev().next(), Some(&1));
    assert_eq!(it.next(), Some(&5));
    assert_eq!(it.by_ref().rev().next(), Some(&2));
    assert_eq!(it.next(), Some(&3));
    assert_eq!(it.next(), None);
    let mut it = xs.iter().rev().skip(5).rev();
    assert_eq!(it.next(), Some(&0));
    assert_eq!(it.rev().next(), Some(&15));
    let mut it_base = xs.iter();
    {
        let mut it = it_base.by_ref().skip(5).rev();
        assert_eq!(it.next(), Some(&30));
        assert_eq!(it.next(), Some(&20));
        assert_eq!(it.next(), Some(&19));
        assert_eq!(it.next(), Some(&17));
        assert_eq!(it.next(), Some(&16));
        assert_eq!(it.next(), Some(&15));
        assert_eq!(it.next(), Some(&13));
        assert_eq!(it.next(), None);
    }
    // make sure the skipped parts have not been consumed
    assert_eq!(it_base.next(), Some(&0));
    assert_eq!(it_base.next(), Some(&1));
    assert_eq!(it_base.next(), Some(&2));
    assert_eq!(it_base.next(), Some(&3));
    assert_eq!(it_base.next(), Some(&5));
    assert_eq!(it_base.next(), None);
    let it = xs.iter().skip(5).rev();
    assert_eq!(it.last(), Some(&13));
}

#[test]
fn test_iterator_skip_nth() {
    let xs = [0, 1, 2, 3, 5, 13, 15, 16, 17, 19, 20, 30];

    let mut it = xs.iter().skip(0);
    assert_eq!(it.nth(0), Some(&0));
    assert_eq!(it.nth(1), Some(&2));

    let mut it = xs.iter().skip(5);
    assert_eq!(it.nth(0), Some(&13));
    assert_eq!(it.nth(1), Some(&16));

    let mut it = xs.iter().skip(12);
    assert_eq!(it.nth(0), None);
}

#[test]
fn test_iterator_skip_count() {
    let xs = [0, 1, 2, 3, 5, 13, 15, 16, 17, 19, 20, 30];

    assert_eq!(xs.iter().skip(0).count(), 12);
    assert_eq!(xs.iter().skip(1).count(), 11);
    assert_eq!(xs.iter().skip(11).count(), 1);
    assert_eq!(xs.iter().skip(12).count(), 0);
    assert_eq!(xs.iter().skip(13).count(), 0);
}

#[test]
fn test_iterator_skip_last() {
    let xs = [0, 1, 2, 3, 5, 13, 15, 16, 17, 19, 20, 30];

    assert_eq!(xs.iter().skip(0).last(), Some(&30));
    assert_eq!(xs.iter().skip(1).last(), Some(&30));
    assert_eq!(xs.iter().skip(11).last(), Some(&30));
    assert_eq!(xs.iter().skip(12).last(), None);
    assert_eq!(xs.iter().skip(13).last(), None);

    let mut it = xs.iter().skip(5);
    assert_eq!(it.next(), Some(&13));
    assert_eq!(it.last(), Some(&30));
}

#[test]
fn test_iterator_skip_fold() {
    let xs = [0, 1, 2, 3, 5, 13, 15, 16, 17, 19, 20, 30];
    let ys = [13, 15, 16, 17, 19, 20, 30];

    let it = xs.iter().skip(5);
    let i = it.fold(0, |i, &x| {
        assert_eq!(x, ys[i]);
        i + 1
    });
    assert_eq!(i, ys.len());

    let mut it = xs.iter().skip(5);
    assert_eq!(it.next(), Some(&ys[0])); // process skips before folding
    let i = it.fold(1, |i, &x| {
        assert_eq!(x, ys[i]);
        i + 1
    });
    assert_eq!(i, ys.len());

    let it = xs.iter().skip(5);
    let i = it.rfold(ys.len(), |i, &x| {
        let i = i - 1;
        assert_eq!(x, ys[i]);
        i
    });
    assert_eq!(i, 0);

    let mut it = xs.iter().skip(5);
    assert_eq!(it.next(), Some(&ys[0])); // process skips before folding
    let i = it.rfold(ys.len(), |i, &x| {
        let i = i - 1;
        assert_eq!(x, ys[i]);
        i
    });
    assert_eq!(i, 1);
}

#[test]
fn test_iterator_take() {
    let xs = [0, 1, 2, 3, 5, 13, 15, 16, 17, 19];
    let ys = [0, 1, 2, 3, 5];

    let mut it = xs.iter().take(ys.len());
    let mut i = 0;
    assert_eq!(it.len(), ys.len());
    while let Some(&x) = it.next() {
        assert_eq!(x, ys[i]);
        i += 1;
        assert_eq!(it.len(), ys.len() - i);
    }
    assert_eq!(i, ys.len());
    assert_eq!(it.len(), 0);

    let mut it = xs.iter().take(ys.len());
    let mut i = 0;
    assert_eq!(it.len(), ys.len());
    while let Some(&x) = it.next_back() {
        i += 1;
        assert_eq!(x, ys[ys.len() - i]);
        assert_eq!(it.len(), ys.len() - i);
    }
    assert_eq!(i, ys.len());
    assert_eq!(it.len(), 0);
}

#[test]
fn test_iterator_take_nth() {
    let xs = [0, 1, 2, 4, 5];
    let mut it = xs.iter();
    {
        let mut take = it.by_ref().take(3);
        let mut i = 0;
        while let Some(&x) = take.nth(0) {
            assert_eq!(x, i);
            i += 1;
        }
    }
    assert_eq!(it.nth(1), Some(&5));
    assert_eq!(it.nth(0), None);

    let xs = [0, 1, 2, 3, 4];
    let mut it = xs.iter().take(7);
    let mut i = 1;
    while let Some(&x) = it.nth(1) {
        assert_eq!(x, i);
        i += 2;
    }
}

#[test]
fn test_iterator_take_nth_back() {
    let xs = [0, 1, 2, 4, 5];
    let mut it = xs.iter();
    {
        let mut take = it.by_ref().take(3);
        let mut i = 0;
        while let Some(&x) = take.nth_back(0) {
            i += 1;
            assert_eq!(x, 3 - i);
        }
    }
    assert_eq!(it.nth_back(0), None);

    let xs = [0, 1, 2, 3, 4];
    let mut it = xs.iter().take(7);
    assert_eq!(it.nth_back(1), Some(&3));
    assert_eq!(it.nth_back(1), Some(&1));
    assert_eq!(it.nth_back(1), None);
}

#[test]
fn test_iterator_take_short() {
    let xs = [0, 1, 2, 3];

    let mut it = xs.iter().take(5);
    let mut i = 0;
    assert_eq!(it.len(), xs.len());
    while let Some(&x) = it.next() {
        assert_eq!(x, xs[i]);
        i += 1;
        assert_eq!(it.len(), xs.len() - i);
    }
    assert_eq!(i, xs.len());
    assert_eq!(it.len(), 0);

    let mut it = xs.iter().take(5);
    let mut i = 0;
    assert_eq!(it.len(), xs.len());
    while let Some(&x) = it.next_back() {
        i += 1;
        assert_eq!(x, xs[xs.len() - i]);
        assert_eq!(it.len(), xs.len() - i);
    }
    assert_eq!(i, xs.len());
    assert_eq!(it.len(), 0);
}

#[test]
fn test_iterator_scan() {
    // test the type inference
    fn add(old: &mut isize, new: &usize) -> Option<f64> {
        *old += *new as isize;
        Some(*old as f64)
    }
    let xs = [0, 1, 2, 3, 4];
    let ys = [0f64, 1.0, 3.0, 6.0, 10.0];

    let it = xs.iter().scan(0, add);
    let mut i = 0;
    for x in it {
        assert_eq!(x, ys[i]);
        i += 1;
    }
    assert_eq!(i, ys.len());
}

#[test]
fn test_iterator_flat_map() {
    let xs = [0, 3, 6];
    let ys = [0, 1, 2, 3, 4, 5, 6, 7, 8];
    let it = xs.iter().flat_map(|&x| (x..).step_by(1).take(3));
    let mut i = 0;
    for x in it {
        assert_eq!(x, ys[i]);
        i += 1;
    }
    assert_eq!(i, ys.len());
}

/// Tests `FlatMap::fold` with items already picked off the front and back,
/// to make sure all parts of the `FlatMap` are folded correctly.
#[test]
fn test_iterator_flat_map_fold() {
    let xs = [0, 3, 6];
    let ys = [1, 2, 3, 4, 5, 6, 7];
    let mut it = xs.iter().flat_map(|&x| x..x + 3);
    assert_eq!(it.next(), Some(0));
    assert_eq!(it.next_back(), Some(8));
    let i = it.fold(0, |i, x| {
        assert_eq!(x, ys[i]);
        i + 1
    });
    assert_eq!(i, ys.len());

    let mut it = xs.iter().flat_map(|&x| x..x + 3);
    assert_eq!(it.next(), Some(0));
    assert_eq!(it.next_back(), Some(8));
    let i = it.rfold(ys.len(), |i, x| {
        assert_eq!(x, ys[i - 1]);
        i - 1
    });
    assert_eq!(i, 0);
}

#[test]
fn test_iterator_flatten() {
    let xs = [0, 3, 6];
    let ys = [0, 1, 2, 3, 4, 5, 6, 7, 8];
    let it = xs.iter().map(|&x| (x..).step_by(1).take(3)).flatten();
    let mut i = 0;
    for x in it {
        assert_eq!(x, ys[i]);
        i += 1;
    }
    assert_eq!(i, ys.len());
}

/// Tests `Flatten::fold` with items already picked off the front and back,
/// to make sure all parts of the `Flatten` are folded correctly.
#[test]
fn test_iterator_flatten_fold() {
    let xs = [0, 3, 6];
    let ys = [1, 2, 3, 4, 5, 6, 7];
    let mut it = xs.iter().map(|&x| x..x + 3).flatten();
    assert_eq!(it.next(), Some(0));
    assert_eq!(it.next_back(), Some(8));
    let i = it.fold(0, |i, x| {
        assert_eq!(x, ys[i]);
        i + 1
    });
    assert_eq!(i, ys.len());

    let mut it = xs.iter().map(|&x| x..x + 3).flatten();
    assert_eq!(it.next(), Some(0));
    assert_eq!(it.next_back(), Some(8));
    let i = it.rfold(ys.len(), |i, x| {
        assert_eq!(x, ys[i - 1]);
        i - 1
    });
    assert_eq!(i, 0);
}

#[test]
fn test_inspect() {
    let xs = [1, 2, 3, 4];
    let mut n = 0;

    let ys = xs.iter().cloned().inspect(|_| n += 1).collect::<Vec<usize>>();

    assert_eq!(n, xs.len());
    assert_eq!(&xs[..], &ys[..]);
}

#[test]
fn test_inspect_fold() {
    let xs = [1, 2, 3, 4];
    let mut n = 0;
    {
        let it = xs.iter().inspect(|_| n += 1);
        let i = it.fold(0, |i, &x| {
            assert_eq!(x, xs[i]);
            i + 1
        });
        assert_eq!(i, xs.len());
    }
    assert_eq!(n, xs.len());

    let mut n = 0;
    {
        let it = xs.iter().inspect(|_| n += 1);
        let i = it.rfold(xs.len(), |i, &x| {
            assert_eq!(x, xs[i - 1]);
            i - 1
        });
        assert_eq!(i, 0);
    }
    assert_eq!(n, xs.len());
}

#[test]
fn test_cycle() {
    let cycle_len = 3;
    let it = (0..).step_by(1).take(cycle_len).cycle();
    assert_eq!(it.size_hint(), (usize::MAX, None));
    for (i, x) in it.take(100).enumerate() {
        assert_eq!(i % cycle_len, x);
    }

    let mut it = (0..).step_by(1).take(0).cycle();
    assert_eq!(it.size_hint(), (0, Some(0)));
    assert_eq!(it.next(), None);

    assert_eq!(empty::<i32>().cycle().fold(0, |acc, x| acc + x), 0);

    assert_eq!(once(1).cycle().skip(1).take(4).fold(0, |acc, x| acc + x), 4);

    assert_eq!((0..10).cycle().take(5).sum::<i32>(), 10);
    assert_eq!((0..10).cycle().take(15).sum::<i32>(), 55);
    assert_eq!((0..10).cycle().take(25).sum::<i32>(), 100);

    let mut iter = (0..10).cycle();
    iter.nth(14);
    assert_eq!(iter.take(8).sum::<i32>(), 38);

    let mut iter = (0..10).cycle();
    iter.nth(9);
    assert_eq!(iter.take(3).sum::<i32>(), 3);
}

#[test]
fn test_iterator_nth() {
    let v: &[_] = &[0, 1, 2, 3, 4];
    for i in 0..v.len() {
        assert_eq!(v.iter().nth(i).unwrap(), &v[i]);
    }
    assert_eq!(v.iter().nth(v.len()), None);
}

#[test]
fn test_iterator_nth_back() {
    let v: &[_] = &[0, 1, 2, 3, 4];
    for i in 0..v.len() {
        assert_eq!(v.iter().nth_back(i).unwrap(), &v[v.len() - 1 - i]);
    }
    assert_eq!(v.iter().nth_back(v.len()), None);
}

#[test]
fn test_iterator_rev_nth_back() {
    let v: &[_] = &[0, 1, 2, 3, 4];
    for i in 0..v.len() {
        assert_eq!(v.iter().rev().nth_back(i).unwrap(), &v[i]);
    }
    assert_eq!(v.iter().rev().nth_back(v.len()), None);
}

#[test]
fn test_iterator_rev_nth() {
    let v: &[_] = &[0, 1, 2, 3, 4];
    for i in 0..v.len() {
        assert_eq!(v.iter().rev().nth(i).unwrap(), &v[v.len() - 1 - i]);
    }
    assert_eq!(v.iter().rev().nth(v.len()), None);
}

#[test]
fn test_iterator_advance_by() {
    let v: &[_] = &[0, 1, 2, 3, 4];

    for i in 0..v.len() {
        let mut iter = v.iter();
        assert_eq!(iter.advance_by(i), Ok(()));
        assert_eq!(iter.next().unwrap(), &v[i]);
        assert_eq!(iter.advance_by(100), Err(v.len() - 1 - i));
    }

    assert_eq!(v.iter().advance_by(v.len()), Ok(()));
    assert_eq!(v.iter().advance_by(100), Err(v.len()));
}

#[test]
fn test_iterator_advance_back_by() {
    let v: &[_] = &[0, 1, 2, 3, 4];

    for i in 0..v.len() {
        let mut iter = v.iter();
        assert_eq!(iter.advance_back_by(i), Ok(()));
        assert_eq!(iter.next_back().unwrap(), &v[v.len() - 1 - i]);
        assert_eq!(iter.advance_back_by(100), Err(v.len() - 1 - i));
    }

    assert_eq!(v.iter().advance_back_by(v.len()), Ok(()));
    assert_eq!(v.iter().advance_back_by(100), Err(v.len()));
}

#[test]
fn test_iterator_rev_advance_by() {
    let v: &[_] = &[0, 1, 2, 3, 4];

    for i in 0..v.len() {
        let mut iter = v.iter().rev();
        assert_eq!(iter.advance_by(i), Ok(()));
        assert_eq!(iter.next().unwrap(), &v[v.len() - 1 - i]);
        assert_eq!(iter.advance_by(100), Err(v.len() - 1 - i));
    }

    assert_eq!(v.iter().rev().advance_by(v.len()), Ok(()));
    assert_eq!(v.iter().rev().advance_by(100), Err(v.len()));
}

#[test]
fn test_iterator_rev_advance_back_by() {
    let v: &[_] = &[0, 1, 2, 3, 4];

    for i in 0..v.len() {
        let mut iter = v.iter().rev();
        assert_eq!(iter.advance_back_by(i), Ok(()));
        assert_eq!(iter.next_back().unwrap(), &v[i]);
        assert_eq!(iter.advance_back_by(100), Err(v.len() - 1 - i));
    }

    assert_eq!(v.iter().rev().advance_back_by(v.len()), Ok(()));
    assert_eq!(v.iter().rev().advance_back_by(100), Err(v.len()));
}

#[test]
fn test_iterator_last() {
    let v: &[_] = &[0, 1, 2, 3, 4];
    assert_eq!(v.iter().last().unwrap(), &4);
    assert_eq!(v[..1].iter().last().unwrap(), &0);
}

#[test]
fn test_iterator_len() {
    let v: &[_] = &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    assert_eq!(v[..4].iter().count(), 4);
    assert_eq!(v[..10].iter().count(), 10);
    assert_eq!(v[..0].iter().count(), 0);
}

#[test]
fn test_iterator_sum() {
    let v: &[i32] = &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    assert_eq!(v[..4].iter().cloned().sum::<i32>(), 6);
    assert_eq!(v.iter().cloned().sum::<i32>(), 55);
    assert_eq!(v[..0].iter().cloned().sum::<i32>(), 0);
}

#[test]
fn test_iterator_sum_result() {
    let v: &[Result<i32, ()>] = &[Ok(1), Ok(2), Ok(3), Ok(4)];
    assert_eq!(v.iter().cloned().sum::<Result<i32, _>>(), Ok(10));
    let v: &[Result<i32, ()>] = &[Ok(1), Err(()), Ok(3), Ok(4)];
    assert_eq!(v.iter().cloned().sum::<Result<i32, _>>(), Err(()));

    #[derive(PartialEq, Debug)]
    struct S(Result<i32, ()>);

    impl Sum<Result<i32, ()>> for S {
        fn sum<I: Iterator<Item = Result<i32, ()>>>(mut iter: I) -> Self {
            // takes the sum by repeatedly calling `next` on `iter`,
            // thus testing that repeated calls to `ResultShunt::try_fold`
            // produce the expected results
            Self(iter.by_ref().sum())
        }
    }

    let v: &[Result<i32, ()>] = &[Ok(1), Ok(2), Ok(3), Ok(4)];
    assert_eq!(v.iter().cloned().sum::<S>(), S(Ok(10)));
    let v: &[Result<i32, ()>] = &[Ok(1), Err(()), Ok(3), Ok(4)];
    assert_eq!(v.iter().cloned().sum::<S>(), S(Err(())));
}

#[test]
fn test_iterator_sum_option() {
    let v: &[Option<i32>] = &[Some(1), Some(2), Some(3), Some(4)];
    assert_eq!(v.iter().cloned().sum::<Option<i32>>(), Some(10));
    let v: &[Option<i32>] = &[Some(1), None, Some(3), Some(4)];
    assert_eq!(v.iter().cloned().sum::<Option<i32>>(), None);
}

#[test]
fn test_iterator_product() {
    let v: &[i32] = &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    assert_eq!(v[..4].iter().cloned().product::<i32>(), 0);
    assert_eq!(v[1..5].iter().cloned().product::<i32>(), 24);
    assert_eq!(v[..0].iter().cloned().product::<i32>(), 1);
}

#[test]
fn test_iterator_product_result() {
    let v: &[Result<i32, ()>] = &[Ok(1), Ok(2), Ok(3), Ok(4)];
    assert_eq!(v.iter().cloned().product::<Result<i32, _>>(), Ok(24));
    let v: &[Result<i32, ()>] = &[Ok(1), Err(()), Ok(3), Ok(4)];
    assert_eq!(v.iter().cloned().product::<Result<i32, _>>(), Err(()));
}

/// A wrapper struct that implements `Eq` and `Ord` based on the wrapped
/// integer modulo 3. Used to test that `Iterator::max` and `Iterator::min`
/// return the correct element if some of them are equal.
#[derive(Debug)]
struct Mod3(i32);

impl PartialEq for Mod3 {
    fn eq(&self, other: &Self) -> bool {
        self.0 % 3 == other.0 % 3
    }
}

impl Eq for Mod3 {}

impl PartialOrd for Mod3 {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Mod3 {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        (self.0 % 3).cmp(&(other.0 % 3))
    }
}

#[test]
fn test_iterator_product_option() {
    let v: &[Option<i32>] = &[Some(1), Some(2), Some(3), Some(4)];
    assert_eq!(v.iter().cloned().product::<Option<i32>>(), Some(24));
    let v: &[Option<i32>] = &[Some(1), None, Some(3), Some(4)];
    assert_eq!(v.iter().cloned().product::<Option<i32>>(), None);
}

#[test]
fn test_iterator_max() {
    let v: &[_] = &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    assert_eq!(v[..4].iter().cloned().max(), Some(3));
    assert_eq!(v.iter().cloned().max(), Some(10));
    assert_eq!(v[..0].iter().cloned().max(), None);
    assert_eq!(v.iter().cloned().map(Mod3).max().map(|x| x.0), Some(8));
}

#[test]
fn test_iterator_min() {
    let v: &[_] = &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    assert_eq!(v[..4].iter().cloned().min(), Some(0));
    assert_eq!(v.iter().cloned().min(), Some(0));
    assert_eq!(v[..0].iter().cloned().min(), None);
    assert_eq!(v.iter().cloned().map(Mod3).min().map(|x| x.0), Some(0));
}

#[test]
fn test_iterator_size_hint() {
    let c = (0..).step_by(1);
    let v: &[_] = &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
    let v2 = &[10, 11, 12];
    let vi = v.iter();

    assert_eq!((0..).size_hint(), (usize::MAX, None));
    assert_eq!(c.size_hint(), (usize::MAX, None));
    assert_eq!(vi.clone().size_hint(), (10, Some(10)));

    assert_eq!(c.clone().take(5).size_hint(), (5, Some(5)));
    assert_eq!(c.clone().skip(5).size_hint().1, None);
    assert_eq!(c.clone().take_while(|_| false).size_hint(), (0, None));
    assert_eq!(c.clone().map_while(|_| None::<()>).size_hint(), (0, None));
    assert_eq!(c.clone().skip_while(|_| false).size_hint(), (0, None));
    assert_eq!(c.clone().enumerate().size_hint(), (usize::MAX, None));
    assert_eq!(c.clone().chain(vi.clone().cloned()).size_hint(), (usize::MAX, None));
    assert_eq!(c.clone().zip(vi.clone()).size_hint(), (10, Some(10)));
    assert_eq!(c.clone().scan(0, |_, _| Some(0)).size_hint(), (0, None));
    assert_eq!(c.clone().filter(|_| false).size_hint(), (0, None));
    assert_eq!(c.clone().map(|_| 0).size_hint(), (usize::MAX, None));
    assert_eq!(c.filter_map(|_| Some(0)).size_hint(), (0, None));

    assert_eq!(vi.clone().take(5).size_hint(), (5, Some(5)));
    assert_eq!(vi.clone().take(12).size_hint(), (10, Some(10)));
    assert_eq!(vi.clone().skip(3).size_hint(), (7, Some(7)));
    assert_eq!(vi.clone().skip(12).size_hint(), (0, Some(0)));
    assert_eq!(vi.clone().take_while(|_| false).size_hint(), (0, Some(10)));
    assert_eq!(vi.clone().map_while(|_| None::<()>).size_hint(), (0, Some(10)));
    assert_eq!(vi.clone().skip_while(|_| false).size_hint(), (0, Some(10)));
    assert_eq!(vi.clone().enumerate().size_hint(), (10, Some(10)));
    assert_eq!(vi.clone().chain(v2).size_hint(), (13, Some(13)));
    assert_eq!(vi.clone().zip(v2).size_hint(), (3, Some(3)));
    assert_eq!(vi.clone().scan(0, |_, _| Some(0)).size_hint(), (0, Some(10)));
    assert_eq!(vi.clone().filter(|_| false).size_hint(), (0, Some(10)));
    assert_eq!(vi.clone().map(|&i| i + 1).size_hint(), (10, Some(10)));
    assert_eq!(vi.filter_map(|_| Some(0)).size_hint(), (0, Some(10)));
}

#[test]
fn test_collect() {
    let a = vec![1, 2, 3, 4, 5];
    let b: Vec<isize> = a.iter().cloned().collect();
    assert!(a == b);
}

#[test]
fn test_all() {
    let v: Box<[isize]> = Box::new([1, 2, 3, 4, 5]);
    assert!(v.iter().all(|&x| x < 10));
    assert!(!v.iter().all(|&x| x % 2 == 0));
    assert!(!v.iter().all(|&x| x > 100));
    assert!(v[..0].iter().all(|_| panic!()));
}

#[test]
fn test_any() {
    let v: Box<[isize]> = Box::new([1, 2, 3, 4, 5]);
    assert!(v.iter().any(|&x| x < 10));
    assert!(v.iter().any(|&x| x % 2 == 0));
    assert!(!v.iter().any(|&x| x > 100));
    assert!(!v[..0].iter().any(|_| panic!()));
}

#[test]
fn test_find() {
    let v: &[isize] = &[1, 3, 9, 27, 103, 14, 11];
    assert_eq!(*v.iter().find(|&&x| x & 1 == 0).unwrap(), 14);
    assert_eq!(*v.iter().find(|&&x| x % 3 == 0).unwrap(), 3);
    assert!(v.iter().find(|&&x| x % 12 == 0).is_none());
}

#[test]
fn test_find_map() {
    let xs: &[isize] = &[];
    assert_eq!(xs.iter().find_map(half_if_even), None);
    let xs: &[isize] = &[3, 5];
    assert_eq!(xs.iter().find_map(half_if_even), None);
    let xs: &[isize] = &[4, 5];
    assert_eq!(xs.iter().find_map(half_if_even), Some(2));
    let xs: &[isize] = &[3, 6];
    assert_eq!(xs.iter().find_map(half_if_even), Some(3));

    let xs: &[isize] = &[1, 2, 3, 4, 5, 6, 7];
    let mut iter = xs.iter();
    assert_eq!(iter.find_map(half_if_even), Some(1));
    assert_eq!(iter.find_map(half_if_even), Some(2));
    assert_eq!(iter.find_map(half_if_even), Some(3));
    assert_eq!(iter.next(), Some(&7));

    fn half_if_even(x: &isize) -> Option<isize> {
        if x % 2 == 0 { Some(x / 2) } else { None }
    }
}

#[test]
fn test_try_find() {
    let xs: &[isize] = &[];
    assert_eq!(xs.iter().try_find(testfn), Ok(None));
    let xs: &[isize] = &[1, 2, 3, 4];
    assert_eq!(xs.iter().try_find(testfn), Ok(Some(&2)));
    let xs: &[isize] = &[1, 3, 4];
    assert_eq!(xs.iter().try_find(testfn), Err(()));

    let xs: &[isize] = &[1, 2, 3, 4, 5, 6, 7];
    let mut iter = xs.iter();
    assert_eq!(iter.try_find(testfn), Ok(Some(&2)));
    assert_eq!(iter.try_find(testfn), Err(()));
    assert_eq!(iter.next(), Some(&5));

    fn testfn(x: &&isize) -> Result<bool, ()> {
        if **x == 2 {
            return Ok(true);
        }
        if **x == 4 {
            return Err(());
        }
        Ok(false)
    }
}

#[test]
fn test_try_find_api_usability() -> Result<(), Box<dyn std::error::Error>> {
    let a = ["1", "2"];

    let is_my_num = |s: &str, search: i32| -> Result<bool, std::num::ParseIntError> {
        Ok(s.parse::<i32>()? == search)
    };

    let val = a.iter().try_find(|&&s| is_my_num(s, 2))?;
    assert_eq!(val, Some(&"2"));

    Ok(())
}

#[test]
fn test_position() {
    let v = &[1, 3, 9, 27, 103, 14, 11];
    assert_eq!(v.iter().position(|x| *x & 1 == 0).unwrap(), 5);
    assert_eq!(v.iter().position(|x| *x % 3 == 0).unwrap(), 1);
    assert!(v.iter().position(|x| *x % 12 == 0).is_none());
}

#[test]
fn test_count() {
    let xs = &[1, 2, 2, 1, 5, 9, 0, 2];
    assert_eq!(xs.iter().filter(|x| **x == 2).count(), 3);
    assert_eq!(xs.iter().filter(|x| **x == 5).count(), 1);
    assert_eq!(xs.iter().filter(|x| **x == 95).count(), 0);
}

#[test]
fn test_max_by_key() {
    let xs: &[isize] = &[-3, 0, 1, 5, -10];
    assert_eq!(*xs.iter().max_by_key(|x| x.abs()).unwrap(), -10);
}

#[test]
fn test_max_by() {
    let xs: &[isize] = &[-3, 0, 1, 5, -10];
    assert_eq!(*xs.iter().max_by(|x, y| x.abs().cmp(&y.abs())).unwrap(), -10);
}

#[test]
fn test_min_by_key() {
    let xs: &[isize] = &[-3, 0, 1, 5, -10];
    assert_eq!(*xs.iter().min_by_key(|x| x.abs()).unwrap(), 0);
}

#[test]
fn test_min_by() {
    let xs: &[isize] = &[-3, 0, 1, 5, -10];
    assert_eq!(*xs.iter().min_by(|x, y| x.abs().cmp(&y.abs())).unwrap(), 0);
}

#[test]
fn test_by_ref() {
    let mut xs = 0..10;
    // sum the first five values
    let partial_sum = xs.by_ref().take(5).fold(0, |a, b| a + b);
    assert_eq!(partial_sum, 10);
    assert_eq!(xs.next(), Some(5));
}

#[test]
fn test_rev() {
    let xs = [2, 4, 6, 8, 10, 12, 14, 16];
    let mut it = xs.iter();
    it.next();
    it.next();
    assert!(it.rev().cloned().collect::<Vec<isize>>() == vec![16, 14, 12, 10, 8, 6]);
}

#[test]
fn test_copied() {
    let xs = [2, 4, 6, 8];

    let mut it = xs.iter().copied();
    assert_eq!(it.len(), 4);
    assert_eq!(it.next(), Some(2));
    assert_eq!(it.len(), 3);
    assert_eq!(it.next(), Some(4));
    assert_eq!(it.len(), 2);
    assert_eq!(it.next_back(), Some(8));
    assert_eq!(it.len(), 1);
    assert_eq!(it.next_back(), Some(6));
    assert_eq!(it.len(), 0);
    assert_eq!(it.next_back(), None);
}

#[test]
fn test_cloned() {
    let xs = [2, 4, 6, 8];

    let mut it = xs.iter().cloned();
    assert_eq!(it.len(), 4);
    assert_eq!(it.next(), Some(2));
    assert_eq!(it.len(), 3);
    assert_eq!(it.next(), Some(4));
    assert_eq!(it.len(), 2);
    assert_eq!(it.next_back(), Some(8));
    assert_eq!(it.len(), 1);
    assert_eq!(it.next_back(), Some(6));
    assert_eq!(it.len(), 0);
    assert_eq!(it.next_back(), None);
}

#[test]
fn test_cloned_side_effects() {
    let mut count = 0;
    {
        let iter = [1, 2, 3]
            .iter()
            .map(|x| {
                count += 1;
                x
            })
            .cloned()
            .zip(&[1]);
        for _ in iter {}
    }
    assert_eq!(count, 2);
}

#[test]
fn test_double_ended_map() {
    let xs = [1, 2, 3, 4, 5, 6];
    let mut it = xs.iter().map(|&x| x * -1);
    assert_eq!(it.next(), Some(-1));
    assert_eq!(it.next(), Some(-2));
    assert_eq!(it.next_back(), Some(-6));
    assert_eq!(it.next_back(), Some(-5));
    assert_eq!(it.next(), Some(-3));
    assert_eq!(it.next_back(), Some(-4));
    assert_eq!(it.next(), None);
}

#[test]
fn test_double_ended_enumerate() {
    let xs = [1, 2, 3, 4, 5, 6];
    let mut it = xs.iter().cloned().enumerate();
    assert_eq!(it.next(), Some((0, 1)));
    assert_eq!(it.next(), Some((1, 2)));
    assert_eq!(it.next_back(), Some((5, 6)));
    assert_eq!(it.next_back(), Some((4, 5)));
    assert_eq!(it.next_back(), Some((3, 4)));
    assert_eq!(it.next_back(), Some((2, 3)));
    assert_eq!(it.next(), None);
}

#[test]
fn test_double_ended_zip() {
    let xs = [1, 2, 3, 4, 5, 6];
    let ys = [1, 2, 3, 7];
    let a = xs.iter().cloned();
    let b = ys.iter().cloned();
    let mut it = a.zip(b);
    assert_eq!(it.next(), Some((1, 1)));
    assert_eq!(it.next(), Some((2, 2)));
    assert_eq!(it.next_back(), Some((4, 7)));
    assert_eq!(it.next_back(), Some((3, 3)));
    assert_eq!(it.next(), None);
}

#[test]
fn test_double_ended_filter() {
    let xs = [1, 2, 3, 4, 5, 6];
    let mut it = xs.iter().filter(|&x| *x & 1 == 0);
    assert_eq!(it.next_back().unwrap(), &6);
    assert_eq!(it.next_back().unwrap(), &4);
    assert_eq!(it.next().unwrap(), &2);
    assert_eq!(it.next_back(), None);
}

#[test]
fn test_double_ended_filter_map() {
    let xs = [1, 2, 3, 4, 5, 6];
    let mut it = xs.iter().filter_map(|&x| if x & 1 == 0 { Some(x * 2) } else { None });
    assert_eq!(it.next_back().unwrap(), 12);
    assert_eq!(it.next_back().unwrap(), 8);
    assert_eq!(it.next().unwrap(), 4);
    assert_eq!(it.next_back(), None);
}

#[test]
fn test_double_ended_chain() {
    let xs = [1, 2, 3, 4, 5];
    let ys = [7, 9, 11];
    let mut it = xs.iter().chain(&ys).rev();
    assert_eq!(it.next().unwrap(), &11);
    assert_eq!(it.next().unwrap(), &9);
    assert_eq!(it.next_back().unwrap(), &1);
    assert_eq!(it.next_back().unwrap(), &2);
    assert_eq!(it.next_back().unwrap(), &3);
    assert_eq!(it.next_back().unwrap(), &4);
    assert_eq!(it.next_back().unwrap(), &5);
    assert_eq!(it.next_back().unwrap(), &7);
    assert_eq!(it.next_back(), None);

    // test that .chain() is well behaved with an unfused iterator
    struct CrazyIterator(bool);
    impl CrazyIterator {
        fn new() -> CrazyIterator {
            CrazyIterator(false)
        }
    }
    impl Iterator for CrazyIterator {
        type Item = i32;
        fn next(&mut self) -> Option<i32> {
            if self.0 {
                Some(99)
            } else {
                self.0 = true;
                None
            }
        }
    }

    impl DoubleEndedIterator for CrazyIterator {
        fn next_back(&mut self) -> Option<i32> {
            self.next()
        }
    }

    assert_eq!(CrazyIterator::new().chain(0..10).rev().last(), Some(0));
    assert!((0..10).chain(CrazyIterator::new()).rev().any(|i| i == 0));
}

#[test]
fn test_rposition() {
    fn f(xy: &(isize, char)) -> bool {
        let (_x, y) = *xy;
        y == 'b'
    }
    fn g(xy: &(isize, char)) -> bool {
        let (_x, y) = *xy;
        y == 'd'
    }
    let v = [(0, 'a'), (1, 'b'), (2, 'c'), (3, 'b')];

    assert_eq!(v.iter().rposition(f), Some(3));
    assert!(v.iter().rposition(g).is_none());
}

#[test]
fn test_rev_rposition() {
    let v = [0, 0, 1, 1];
    assert_eq!(v.iter().rev().rposition(|&x| x == 1), Some(1));
}

#[test]
#[should_panic]
fn test_rposition_panic() {
    let v: [(Box<_>, Box<_>); 4] = [(box 0, box 0), (box 0, box 0), (box 0, box 0), (box 0, box 0)];
    let mut i = 0;
    v.iter().rposition(|_elt| {
        if i == 2 {
            panic!()
        }
        i += 1;
        false
    });
}

#[test]
fn test_double_ended_flat_map() {
    let u = [0, 1];
    let v = [5, 6, 7, 8];
    let mut it = u.iter().flat_map(|x| &v[*x..v.len()]);
    assert_eq!(it.next_back().unwrap(), &8);
    assert_eq!(it.next().unwrap(), &5);
    assert_eq!(it.next_back().unwrap(), &7);
    assert_eq!(it.next_back().unwrap(), &6);
    assert_eq!(it.next_back().unwrap(), &8);
    assert_eq!(it.next().unwrap(), &6);
    assert_eq!(it.next_back().unwrap(), &7);
    assert_eq!(it.next_back(), None);
    assert_eq!(it.next(), None);
    assert_eq!(it.next_back(), None);
}

#[test]
fn test_double_ended_flatten() {
    let u = [0, 1];
    let v = [5, 6, 7, 8];
    let mut it = u.iter().map(|x| &v[*x..v.len()]).flatten();
    assert_eq!(it.next_back().unwrap(), &8);
    assert_eq!(it.next().unwrap(), &5);
    assert_eq!(it.next_back().unwrap(), &7);
    assert_eq!(it.next_back().unwrap(), &6);
    assert_eq!(it.next_back().unwrap(), &8);
    assert_eq!(it.next().unwrap(), &6);
    assert_eq!(it.next_back().unwrap(), &7);
    assert_eq!(it.next_back(), None);
    assert_eq!(it.next(), None);
    assert_eq!(it.next_back(), None);
}

#[test]
fn test_double_ended_range() {
    assert_eq!((11..14).rev().collect::<Vec<_>>(), [13, 12, 11]);
    for _ in (10..0).rev() {
        panic!("unreachable");
    }

    assert_eq!((11..14).rev().collect::<Vec<_>>(), [13, 12, 11]);
    for _ in (10..0).rev() {
        panic!("unreachable");
    }
}

#[test]
fn test_range() {
    assert_eq!((0..5).collect::<Vec<_>>(), [0, 1, 2, 3, 4]);
    assert_eq!((-10..-1).collect::<Vec<_>>(), [-10, -9, -8, -7, -6, -5, -4, -3, -2]);
    assert_eq!((0..5).rev().collect::<Vec<_>>(), [4, 3, 2, 1, 0]);
    assert_eq!((200..-5).count(), 0);
    assert_eq!((200..-5).rev().count(), 0);
    assert_eq!((200..200).count(), 0);
    assert_eq!((200..200).rev().count(), 0);

    assert_eq!((0..100).size_hint(), (100, Some(100)));
    // this test is only meaningful when sizeof usize < sizeof u64
    assert_eq!((usize::MAX - 1..usize::MAX).size_hint(), (1, Some(1)));
    assert_eq!((-10..-1).size_hint(), (9, Some(9)));
    assert_eq!((-1..-10).size_hint(), (0, Some(0)));

    assert_eq!((-70..58).size_hint(), (128, Some(128)));
    assert_eq!((-128..127).size_hint(), (255, Some(255)));
    assert_eq!(
        (-2..isize::MAX).size_hint(),
        (isize::MAX as usize + 2, Some(isize::MAX as usize + 2))
    );
}

#[test]
fn test_char_range() {
    use std::char;
    // Miri is too slow
    let from = if cfg!(miri) { char::from_u32(0xD800 - 10).unwrap() } else { '\0' };
    let to = if cfg!(miri) { char::from_u32(0xDFFF + 10).unwrap() } else { char::MAX };
    assert!((from..=to).eq((from as u32..=to as u32).filter_map(char::from_u32)));
    assert!((from..=to).rev().eq((from as u32..=to as u32).filter_map(char::from_u32).rev()));

    assert_eq!(('\u{D7FF}'..='\u{E000}').count(), 2);
    assert_eq!(('\u{D7FF}'..='\u{E000}').size_hint(), (2, Some(2)));
    assert_eq!(('\u{D7FF}'..'\u{E000}').count(), 1);
    assert_eq!(('\u{D7FF}'..'\u{E000}').size_hint(), (1, Some(1)));
}

#[test]
fn test_range_exhaustion() {
    let mut r = 10..10;
    assert!(r.is_empty());
    assert_eq!(r.next(), None);
    assert_eq!(r.next_back(), None);
    assert_eq!(r, 10..10);

    let mut r = 10..12;
    assert_eq!(r.next(), Some(10));
    assert_eq!(r.next(), Some(11));
    assert!(r.is_empty());
    assert_eq!(r, 12..12);
    assert_eq!(r.next(), None);

    let mut r = 10..12;
    assert_eq!(r.next_back(), Some(11));
    assert_eq!(r.next_back(), Some(10));
    assert!(r.is_empty());
    assert_eq!(r, 10..10);
    assert_eq!(r.next_back(), None);

    let mut r = 100..10;
    assert!(r.is_empty());
    assert_eq!(r.next(), None);
    assert_eq!(r.next_back(), None);
    assert_eq!(r, 100..10);
}

#[test]
fn test_range_inclusive_exhaustion() {
    let mut r = 10..=10;
    assert_eq!(r.next(), Some(10));
    assert!(r.is_empty());
    assert_eq!(r.next(), None);
    assert_eq!(r.next(), None);

    assert_eq!(*r.start(), 10);
    assert_eq!(*r.end(), 10);
    assert_ne!(r, 10..=10);

    let mut r = 10..=10;
    assert_eq!(r.next_back(), Some(10));
    assert!(r.is_empty());
    assert_eq!(r.next_back(), None);

    assert_eq!(*r.start(), 10);
    assert_eq!(*r.end(), 10);
    assert_ne!(r, 10..=10);

    let mut r = 10..=12;
    assert_eq!(r.next(), Some(10));
    assert_eq!(r.next(), Some(11));
    assert_eq!(r.next(), Some(12));
    assert!(r.is_empty());
    assert_eq!(r.next(), None);

    let mut r = 10..=12;
    assert_eq!(r.next_back(), Some(12));
    assert_eq!(r.next_back(), Some(11));
    assert_eq!(r.next_back(), Some(10));
    assert!(r.is_empty());
    assert_eq!(r.next_back(), None);

    let mut r = 10..=12;
    assert_eq!(r.nth(2), Some(12));
    assert!(r.is_empty());
    assert_eq!(r.next(), None);

    let mut r = 10..=12;
    assert_eq!(r.nth(5), None);
    assert!(r.is_empty());
    assert_eq!(r.next(), None);

    let mut r = 100..=10;
    assert_eq!(r.next(), None);
    assert!(r.is_empty());
    assert_eq!(r.next(), None);
    assert_eq!(r.next(), None);
    assert_eq!(r, 100..=10);

    let mut r = 100..=10;
    assert_eq!(r.next_back(), None);
    assert!(r.is_empty());
    assert_eq!(r.next_back(), None);
    assert_eq!(r.next_back(), None);
    assert_eq!(r, 100..=10);
}

#[test]
fn test_range_nth() {
    assert_eq!((10..15).nth(0), Some(10));
    assert_eq!((10..15).nth(1), Some(11));
    assert_eq!((10..15).nth(4), Some(14));
    assert_eq!((10..15).nth(5), None);

    let mut r = 10..20;
    assert_eq!(r.nth(2), Some(12));
    assert_eq!(r, 13..20);
    assert_eq!(r.nth(2), Some(15));
    assert_eq!(r, 16..20);
    assert_eq!(r.nth(10), None);
    assert_eq!(r, 20..20);
}

#[test]
fn test_range_nth_back() {
    assert_eq!((10..15).nth_back(0), Some(14));
    assert_eq!((10..15).nth_back(1), Some(13));
    assert_eq!((10..15).nth_back(4), Some(10));
    assert_eq!((10..15).nth_back(5), None);
    assert_eq!((-120..80_i8).nth_back(199), Some(-120));

    let mut r = 10..20;
    assert_eq!(r.nth_back(2), Some(17));
    assert_eq!(r, 10..17);
    assert_eq!(r.nth_back(2), Some(14));
    assert_eq!(r, 10..14);
    assert_eq!(r.nth_back(10), None);
    assert_eq!(r, 10..10);
}

#[test]
fn test_range_from_nth() {
    assert_eq!((10..).nth(0), Some(10));
    assert_eq!((10..).nth(1), Some(11));
    assert_eq!((10..).nth(4), Some(14));

    let mut r = 10..;
    assert_eq!(r.nth(2), Some(12));
    assert_eq!(r, 13..);
    assert_eq!(r.nth(2), Some(15));
    assert_eq!(r, 16..);
    assert_eq!(r.nth(10), Some(26));
    assert_eq!(r, 27..);

    assert_eq!((0..).size_hint(), (usize::MAX, None));
}

fn is_trusted_len<I: TrustedLen>(_: I) {}

#[test]
fn test_range_from_take() {
    let mut it = (0..).take(3);
    assert_eq!(it.next(), Some(0));
    assert_eq!(it.next(), Some(1));
    assert_eq!(it.next(), Some(2));
    assert_eq!(it.next(), None);
    is_trusted_len((0..).take(3));
    assert_eq!((0..).take(3).size_hint(), (3, Some(3)));
    assert_eq!((0..).take(0).size_hint(), (0, Some(0)));
    assert_eq!((0..).take(usize::MAX).size_hint(), (usize::MAX, Some(usize::MAX)));
}

#[test]
fn test_range_from_take_collect() {
    let v: Vec<_> = (0..).take(3).collect();
    assert_eq!(v, vec![0, 1, 2]);
}

#[test]
fn test_range_inclusive_nth() {
    assert_eq!((10..=15).nth(0), Some(10));
    assert_eq!((10..=15).nth(1), Some(11));
    assert_eq!((10..=15).nth(5), Some(15));
    assert_eq!((10..=15).nth(6), None);

    let mut exhausted_via_next = 10_u8..=20;
    while exhausted_via_next.next().is_some() {}

    let mut r = 10_u8..=20;
    assert_eq!(r.nth(2), Some(12));
    assert_eq!(r, 13..=20);
    assert_eq!(r.nth(2), Some(15));
    assert_eq!(r, 16..=20);
    assert_eq!(r.is_empty(), false);
    assert_eq!(ExactSizeIterator::is_empty(&r), false);
    assert_eq!(r.nth(10), None);
    assert_eq!(r.is_empty(), true);
    assert_eq!(r, exhausted_via_next);
    assert_eq!(ExactSizeIterator::is_empty(&r), true);
}

#[test]
fn test_range_inclusive_nth_back() {
    assert_eq!((10..=15).nth_back(0), Some(15));
    assert_eq!((10..=15).nth_back(1), Some(14));
    assert_eq!((10..=15).nth_back(5), Some(10));
    assert_eq!((10..=15).nth_back(6), None);
    assert_eq!((-120..=80_i8).nth_back(200), Some(-120));

    let mut exhausted_via_next_back = 10_u8..=20;
    while exhausted_via_next_back.next_back().is_some() {}

    let mut r = 10_u8..=20;
    assert_eq!(r.nth_back(2), Some(18));
    assert_eq!(r, 10..=17);
    assert_eq!(r.nth_back(2), Some(15));
    assert_eq!(r, 10..=14);
    assert_eq!(r.is_empty(), false);
    assert_eq!(ExactSizeIterator::is_empty(&r), false);
    assert_eq!(r.nth_back(10), None);
    assert_eq!(r.is_empty(), true);
    assert_eq!(r, exhausted_via_next_back);
    assert_eq!(ExactSizeIterator::is_empty(&r), true);
}

#[test]
fn test_range_len() {
    assert_eq!((0..10_u8).len(), 10);
    assert_eq!((9..10_u8).len(), 1);
    assert_eq!((10..10_u8).len(), 0);
    assert_eq!((11..10_u8).len(), 0);
    assert_eq!((100..10_u8).len(), 0);
}

#[test]
fn test_range_inclusive_len() {
    assert_eq!((0..=10_u8).len(), 11);
    assert_eq!((9..=10_u8).len(), 2);
    assert_eq!((10..=10_u8).len(), 1);
    assert_eq!((11..=10_u8).len(), 0);
    assert_eq!((100..=10_u8).len(), 0);
}

#[test]
fn test_range_step() {
    #![allow(deprecated)]

    assert_eq!((0..20).step_by(5).collect::<Vec<isize>>(), [0, 5, 10, 15]);
    assert_eq!((1..21).rev().step_by(5).collect::<Vec<isize>>(), [20, 15, 10, 5]);
    assert_eq!((1..21).rev().step_by(6).collect::<Vec<isize>>(), [20, 14, 8, 2]);
    assert_eq!((200..255).step_by(50).collect::<Vec<u8>>(), [200, 250]);
    assert_eq!((200..-5).step_by(1).collect::<Vec<isize>>(), []);
    assert_eq!((200..200).step_by(1).collect::<Vec<isize>>(), []);

    assert_eq!((0..20).step_by(1).size_hint(), (20, Some(20)));
    assert_eq!((0..20).step_by(21).size_hint(), (1, Some(1)));
    assert_eq!((0..20).step_by(5).size_hint(), (4, Some(4)));
    assert_eq!((1..21).rev().step_by(5).size_hint(), (4, Some(4)));
    assert_eq!((1..21).rev().step_by(6).size_hint(), (4, Some(4)));
    assert_eq!((20..-5).step_by(1).size_hint(), (0, Some(0)));
    assert_eq!((20..20).step_by(1).size_hint(), (0, Some(0)));
    assert_eq!((i8::MIN..i8::MAX).step_by(-(i8::MIN as i32) as usize).size_hint(), (2, Some(2)));
    assert_eq!((i16::MIN..i16::MAX).step_by(i16::MAX as usize).size_hint(), (3, Some(3)));
    assert_eq!((isize::MIN..isize::MAX).step_by(1).size_hint(), (usize::MAX, Some(usize::MAX)));
}

#[test]
fn test_step_by_skip() {
    assert_eq!((0..640).step_by(128).skip(1).collect::<Vec<_>>(), [128, 256, 384, 512]);
    assert_eq!((0..=50).step_by(10).nth(3), Some(30));
    assert_eq!((200..=255u8).step_by(10).nth(3), Some(230));
}

#[test]
fn test_range_inclusive_step() {
    assert_eq!((0..=50).step_by(10).collect::<Vec<_>>(), [0, 10, 20, 30, 40, 50]);
    assert_eq!((0..=5).step_by(1).collect::<Vec<_>>(), [0, 1, 2, 3, 4, 5]);
    assert_eq!((200..=255u8).step_by(10).collect::<Vec<_>>(), [200, 210, 220, 230, 240, 250]);
    assert_eq!((250..=255u8).step_by(1).collect::<Vec<_>>(), [250, 251, 252, 253, 254, 255]);
}

#[test]
fn test_range_last_max() {
    assert_eq!((0..20).last(), Some(19));
    assert_eq!((-20..0).last(), Some(-1));
    assert_eq!((5..5).last(), None);

    assert_eq!((0..20).max(), Some(19));
    assert_eq!((-20..0).max(), Some(-1));
    assert_eq!((5..5).max(), None);
}

#[test]
fn test_range_inclusive_last_max() {
    assert_eq!((0..=20).last(), Some(20));
    assert_eq!((-20..=0).last(), Some(0));
    assert_eq!((5..=5).last(), Some(5));
    let mut r = 10..=10;
    r.next();
    assert_eq!(r.last(), None);

    assert_eq!((0..=20).max(), Some(20));
    assert_eq!((-20..=0).max(), Some(0));
    assert_eq!((5..=5).max(), Some(5));
    let mut r = 10..=10;
    r.next();
    assert_eq!(r.max(), None);
}

#[test]
fn test_range_min() {
    assert_eq!((0..20).min(), Some(0));
    assert_eq!((-20..0).min(), Some(-20));
    assert_eq!((5..5).min(), None);
}

#[test]
fn test_range_inclusive_min() {
    assert_eq!((0..=20).min(), Some(0));
    assert_eq!((-20..=0).min(), Some(-20));
    assert_eq!((5..=5).min(), Some(5));
    let mut r = 10..=10;
    r.next();
    assert_eq!(r.min(), None);
}

#[test]
fn test_range_inclusive_folds() {
    assert_eq!((1..=10).sum::<i32>(), 55);
    assert_eq!((1..=10).rev().sum::<i32>(), 55);

    let mut it = 44..=50;
    assert_eq!(it.try_fold(0, i8::checked_add), None);
    assert_eq!(it, 47..=50);
    assert_eq!(it.try_fold(0, i8::checked_add), None);
    assert_eq!(it, 50..=50);
    assert_eq!(it.try_fold(0, i8::checked_add), Some(50));
    assert!(it.is_empty());
    assert_eq!(it.try_fold(0, i8::checked_add), Some(0));
    assert!(it.is_empty());

    let mut it = 40..=47;
    assert_eq!(it.try_rfold(0, i8::checked_add), None);
    assert_eq!(it, 40..=44);
    assert_eq!(it.try_rfold(0, i8::checked_add), None);
    assert_eq!(it, 40..=41);
    assert_eq!(it.try_rfold(0, i8::checked_add), Some(81));
    assert!(it.is_empty());
    assert_eq!(it.try_rfold(0, i8::checked_add), Some(0));
    assert!(it.is_empty());

    let mut it = 10..=20;
    assert_eq!(it.try_fold(0, |a, b| Some(a + b)), Some(165));
    assert!(it.is_empty());
    assert_eq!(it.try_fold(0, |a, b| Some(a + b)), Some(0));
    assert!(it.is_empty());

    let mut it = 10..=20;
    assert_eq!(it.try_rfold(0, |a, b| Some(a + b)), Some(165));
    assert!(it.is_empty());
    assert_eq!(it.try_rfold(0, |a, b| Some(a + b)), Some(0));
    assert!(it.is_empty());
}

#[test]
fn test_range_size_hint() {
    assert_eq!((0..0usize).size_hint(), (0, Some(0)));
    assert_eq!((0..100usize).size_hint(), (100, Some(100)));
    assert_eq!((0..usize::MAX).size_hint(), (usize::MAX, Some(usize::MAX)));

    let umax = u128::try_from(usize::MAX).unwrap();
    assert_eq!((0..0u128).size_hint(), (0, Some(0)));
    assert_eq!((0..100u128).size_hint(), (100, Some(100)));
    assert_eq!((0..umax).size_hint(), (usize::MAX, Some(usize::MAX)));
    assert_eq!((0..umax + 1).size_hint(), (usize::MAX, None));

    assert_eq!((0..0isize).size_hint(), (0, Some(0)));
    assert_eq!((-100..100isize).size_hint(), (200, Some(200)));
    assert_eq!((isize::MIN..isize::MAX).size_hint(), (usize::MAX, Some(usize::MAX)));

    let imin = i128::try_from(isize::MIN).unwrap();
    let imax = i128::try_from(isize::MAX).unwrap();
    assert_eq!((0..0i128).size_hint(), (0, Some(0)));
    assert_eq!((-100..100i128).size_hint(), (200, Some(200)));
    assert_eq!((imin..imax).size_hint(), (usize::MAX, Some(usize::MAX)));
    assert_eq!((imin..imax + 1).size_hint(), (usize::MAX, None));
}

#[test]
fn test_range_inclusive_size_hint() {
    assert_eq!((1..=0usize).size_hint(), (0, Some(0)));
    assert_eq!((0..=0usize).size_hint(), (1, Some(1)));
    assert_eq!((0..=100usize).size_hint(), (101, Some(101)));
    assert_eq!((0..=usize::MAX - 1).size_hint(), (usize::MAX, Some(usize::MAX)));
    assert_eq!((0..=usize::MAX).size_hint(), (usize::MAX, None));

    let umax = u128::try_from(usize::MAX).unwrap();
    assert_eq!((1..=0u128).size_hint(), (0, Some(0)));
    assert_eq!((0..=0u128).size_hint(), (1, Some(1)));
    assert_eq!((0..=100u128).size_hint(), (101, Some(101)));
    assert_eq!((0..=umax - 1).size_hint(), (usize::MAX, Some(usize::MAX)));
    assert_eq!((0..=umax).size_hint(), (usize::MAX, None));
    assert_eq!((0..=umax + 1).size_hint(), (usize::MAX, None));

    assert_eq!((0..=-1isize).size_hint(), (0, Some(0)));
    assert_eq!((0..=0isize).size_hint(), (1, Some(1)));
    assert_eq!((-100..=100isize).size_hint(), (201, Some(201)));
    assert_eq!((isize::MIN..=isize::MAX - 1).size_hint(), (usize::MAX, Some(usize::MAX)));
    assert_eq!((isize::MIN..=isize::MAX).size_hint(), (usize::MAX, None));

    let imin = i128::try_from(isize::MIN).unwrap();
    let imax = i128::try_from(isize::MAX).unwrap();
    assert_eq!((0..=-1i128).size_hint(), (0, Some(0)));
    assert_eq!((0..=0i128).size_hint(), (1, Some(1)));
    assert_eq!((-100..=100i128).size_hint(), (201, Some(201)));
    assert_eq!((imin..=imax - 1).size_hint(), (usize::MAX, Some(usize::MAX)));
    assert_eq!((imin..=imax).size_hint(), (usize::MAX, None));
    assert_eq!((imin..=imax + 1).size_hint(), (usize::MAX, None));
}

#[test]
fn test_repeat() {
    let mut it = repeat(42);
    assert_eq!(it.next(), Some(42));
    assert_eq!(it.next(), Some(42));
    assert_eq!(it.next(), Some(42));
    assert_eq!(repeat(42).size_hint(), (usize::MAX, None));
}

#[test]
fn test_repeat_take() {
    let mut it = repeat(42).take(3);
    assert_eq!(it.next(), Some(42));
    assert_eq!(it.next(), Some(42));
    assert_eq!(it.next(), Some(42));
    assert_eq!(it.next(), None);
    is_trusted_len(repeat(42).take(3));
    assert_eq!(repeat(42).take(3).size_hint(), (3, Some(3)));
    assert_eq!(repeat(42).take(0).size_hint(), (0, Some(0)));
    assert_eq!(repeat(42).take(usize::MAX).size_hint(), (usize::MAX, Some(usize::MAX)));
}

#[test]
fn test_repeat_take_collect() {
    let v: Vec<_> = repeat(42).take(3).collect();
    assert_eq!(v, vec![42, 42, 42]);
}

#[test]
fn test_repeat_with() {
    #[derive(PartialEq, Debug)]
    struct NotClone(usize);
    let mut it = repeat_with(|| NotClone(42));
    assert_eq!(it.next(), Some(NotClone(42)));
    assert_eq!(it.next(), Some(NotClone(42)));
    assert_eq!(it.next(), Some(NotClone(42)));
    assert_eq!(repeat_with(|| NotClone(42)).size_hint(), (usize::MAX, None));
}

#[test]
fn test_repeat_with_take() {
    let mut it = repeat_with(|| 42).take(3);
    assert_eq!(it.next(), Some(42));
    assert_eq!(it.next(), Some(42));
    assert_eq!(it.next(), Some(42));
    assert_eq!(it.next(), None);
    is_trusted_len(repeat_with(|| 42).take(3));
    assert_eq!(repeat_with(|| 42).take(3).size_hint(), (3, Some(3)));
    assert_eq!(repeat_with(|| 42).take(0).size_hint(), (0, Some(0)));
    assert_eq!(repeat_with(|| 42).take(usize::MAX).size_hint(), (usize::MAX, Some(usize::MAX)));
}

#[test]
fn test_repeat_with_take_collect() {
    let mut curr = 1;
    let v: Vec<_> = repeat_with(|| {
        let tmp = curr;
        curr *= 2;
        tmp
    })
    .take(5)
    .collect();
    assert_eq!(v, vec![1, 2, 4, 8, 16]);
}

#[test]
fn test_successors() {
    let mut powers_of_10 = successors(Some(1_u16), |n| n.checked_mul(10));
    assert_eq!(powers_of_10.by_ref().collect::<Vec<_>>(), &[1, 10, 100, 1_000, 10_000]);
    assert_eq!(powers_of_10.next(), None);

    let mut empty = successors(None::<u32>, |_| unimplemented!());
    assert_eq!(empty.next(), None);
    assert_eq!(empty.next(), None);
}

#[test]
fn test_fuse() {
    let mut it = 0..3;
    assert_eq!(it.len(), 3);
    assert_eq!(it.next(), Some(0));
    assert_eq!(it.len(), 2);
    assert_eq!(it.next(), Some(1));
    assert_eq!(it.len(), 1);
    assert_eq!(it.next(), Some(2));
    assert_eq!(it.len(), 0);
    assert_eq!(it.next(), None);
    assert_eq!(it.len(), 0);
    assert_eq!(it.next(), None);
    assert_eq!(it.len(), 0);
    assert_eq!(it.next(), None);
    assert_eq!(it.len(), 0);
}

#[test]
fn test_fuse_nth() {
    let xs = [0, 1, 2];
    let mut it = xs.iter();

    assert_eq!(it.len(), 3);
    assert_eq!(it.nth(2), Some(&2));
    assert_eq!(it.len(), 0);
    assert_eq!(it.nth(2), None);
    assert_eq!(it.len(), 0);
}

#[test]
fn test_fuse_last() {
    let xs = [0, 1, 2];
    let it = xs.iter();

    assert_eq!(it.len(), 3);
    assert_eq!(it.last(), Some(&2));
}

#[test]
fn test_fuse_count() {
    let xs = [0, 1, 2];
    let it = xs.iter();

    assert_eq!(it.len(), 3);
    assert_eq!(it.count(), 3);
    // Can't check len now because count consumes.
}

#[test]
fn test_fuse_fold() {
    let xs = [0, 1, 2];
    let it = xs.iter(); // `FusedIterator`
    let i = it.fuse().fold(0, |i, &x| {
        assert_eq!(x, xs[i]);
        i + 1
    });
    assert_eq!(i, xs.len());

    let it = xs.iter(); // `FusedIterator`
    let i = it.fuse().rfold(xs.len(), |i, &x| {
        assert_eq!(x, xs[i - 1]);
        i - 1
    });
    assert_eq!(i, 0);

    let it = xs.iter().scan((), |_, &x| Some(x)); // `!FusedIterator`
    let i = it.fuse().fold(0, |i, x| {
        assert_eq!(x, xs[i]);
        i + 1
    });
    assert_eq!(i, xs.len());
}

#[test]
fn test_once() {
    let mut it = once(42);
    assert_eq!(it.next(), Some(42));
    assert_eq!(it.next(), None);
}

#[test]
fn test_once_with() {
    let count = Cell::new(0);
    let mut it = once_with(|| {
        count.set(count.get() + 1);
        42
    });

    assert_eq!(count.get(), 0);
    assert_eq!(it.next(), Some(42));
    assert_eq!(count.get(), 1);
    assert_eq!(it.next(), None);
    assert_eq!(count.get(), 1);
    assert_eq!(it.next(), None);
    assert_eq!(count.get(), 1);
}

#[test]
fn test_empty() {
    let mut it = empty::<i32>();
    assert_eq!(it.next(), None);
}

#[test]
fn test_chain_fold() {
    let xs = [1, 2, 3];
    let ys = [1, 2, 0];

    let mut iter = xs.iter().chain(&ys);
    iter.next();
    let mut result = Vec::new();
    iter.fold((), |(), &elt| result.push(elt));
    assert_eq!(&[2, 3, 1, 2, 0], &result[..]);
}

#[test]
fn test_steps_between() {
    assert_eq!(Step::steps_between(&20_u8, &200_u8), Some(180_usize));
    assert_eq!(Step::steps_between(&-20_i8, &80_i8), Some(100_usize));
    assert_eq!(Step::steps_between(&-120_i8, &80_i8), Some(200_usize));
    assert_eq!(Step::steps_between(&20_u32, &4_000_100_u32), Some(4_000_080_usize));
    assert_eq!(Step::steps_between(&-20_i32, &80_i32), Some(100_usize));
    assert_eq!(Step::steps_between(&-2_000_030_i32, &2_000_050_i32), Some(4_000_080_usize));

    // Skip u64/i64 to avoid differences with 32-bit vs 64-bit platforms

    assert_eq!(Step::steps_between(&20_u128, &200_u128), Some(180_usize));
    assert_eq!(Step::steps_between(&-20_i128, &80_i128), Some(100_usize));
    if cfg!(target_pointer_width = "64") {
        assert_eq!(Step::steps_between(&10_u128, &0x1_0000_0000_0000_0009_u128), Some(usize::MAX));
    }
    assert_eq!(Step::steps_between(&10_u128, &0x1_0000_0000_0000_000a_u128), None);
    assert_eq!(Step::steps_between(&10_i128, &0x1_0000_0000_0000_000a_i128), None);
    assert_eq!(
        Step::steps_between(&-0x1_0000_0000_0000_0000_i128, &0x1_0000_0000_0000_0000_i128,),
        None,
    );
}

#[test]
fn test_step_forward() {
    assert_eq!(Step::forward_checked(55_u8, 200_usize), Some(255_u8));
    assert_eq!(Step::forward_checked(252_u8, 200_usize), None);
    assert_eq!(Step::forward_checked(0_u8, 256_usize), None);
    assert_eq!(Step::forward_checked(-110_i8, 200_usize), Some(90_i8));
    assert_eq!(Step::forward_checked(-110_i8, 248_usize), None);
    assert_eq!(Step::forward_checked(-126_i8, 256_usize), None);

    assert_eq!(Step::forward_checked(35_u16, 100_usize), Some(135_u16));
    assert_eq!(Step::forward_checked(35_u16, 65500_usize), Some(u16::MAX));
    assert_eq!(Step::forward_checked(36_u16, 65500_usize), None);
    assert_eq!(Step::forward_checked(-110_i16, 200_usize), Some(90_i16));
    assert_eq!(Step::forward_checked(-20_030_i16, 50_050_usize), Some(30_020_i16));
    assert_eq!(Step::forward_checked(-10_i16, 40_000_usize), None);
    assert_eq!(Step::forward_checked(-10_i16, 70_000_usize), None);

    assert_eq!(Step::forward_checked(10_u128, 70_000_usize), Some(70_010_u128));
    assert_eq!(Step::forward_checked(10_i128, 70_030_usize), Some(70_040_i128));
    assert_eq!(
        Step::forward_checked(0xffff_ffff_ffff_ffff__ffff_ffff_ffff_ff00_u128, 0xff_usize),
        Some(u128::MAX),
    );
    assert_eq!(
        Step::forward_checked(0xffff_ffff_ffff_ffff__ffff_ffff_ffff_ff00_u128, 0x100_usize),
        None
    );
    assert_eq!(
        Step::forward_checked(0x7fff_ffff_ffff_ffff__ffff_ffff_ffff_ff00_i128, 0xff_usize),
        Some(i128::MAX),
    );
    assert_eq!(
        Step::forward_checked(0x7fff_ffff_ffff_ffff__ffff_ffff_ffff_ff00_i128, 0x100_usize),
        None
    );
}

#[test]
fn test_step_backward() {
    assert_eq!(Step::backward_checked(255_u8, 200_usize), Some(55_u8));
    assert_eq!(Step::backward_checked(100_u8, 200_usize), None);
    assert_eq!(Step::backward_checked(255_u8, 256_usize), None);
    assert_eq!(Step::backward_checked(90_i8, 200_usize), Some(-110_i8));
    assert_eq!(Step::backward_checked(110_i8, 248_usize), None);
    assert_eq!(Step::backward_checked(127_i8, 256_usize), None);

    assert_eq!(Step::backward_checked(135_u16, 100_usize), Some(35_u16));
    assert_eq!(Step::backward_checked(u16::MAX, 65500_usize), Some(35_u16));
    assert_eq!(Step::backward_checked(10_u16, 11_usize), None);
    assert_eq!(Step::backward_checked(90_i16, 200_usize), Some(-110_i16));
    assert_eq!(Step::backward_checked(30_020_i16, 50_050_usize), Some(-20_030_i16));
    assert_eq!(Step::backward_checked(-10_i16, 40_000_usize), None);
    assert_eq!(Step::backward_checked(-10_i16, 70_000_usize), None);

    assert_eq!(Step::backward_checked(70_010_u128, 70_000_usize), Some(10_u128));
    assert_eq!(Step::backward_checked(70_020_i128, 70_030_usize), Some(-10_i128));
    assert_eq!(Step::backward_checked(10_u128, 7_usize), Some(3_u128));
    assert_eq!(Step::backward_checked(10_u128, 11_usize), None);
    assert_eq!(
        Step::backward_checked(-0x7fff_ffff_ffff_ffff__ffff_ffff_ffff_ff00_i128, 0x100_usize),
        Some(i128::MIN)
    );
}

#[test]
fn test_rev_try_folds() {
    let f = &|acc, x| i32::checked_add(2 * acc, x);
    assert_eq!((1..10).rev().try_fold(7, f), (1..10).try_rfold(7, f));
    assert_eq!((1..10).rev().try_rfold(7, f), (1..10).try_fold(7, f));

    let a = [10, 20, 30, 40, 100, 60, 70, 80, 90];
    let mut iter = a.iter().rev();
    assert_eq!(iter.try_fold(0_i8, |acc, &x| acc.checked_add(x)), None);
    assert_eq!(iter.next(), Some(&70));
    let mut iter = a.iter().rev();
    assert_eq!(iter.try_rfold(0_i8, |acc, &x| acc.checked_add(x)), None);
    assert_eq!(iter.next_back(), Some(&60));
}

#[test]
fn test_cloned_try_folds() {
    let a = [1, 2, 3, 4, 5, 6, 7, 8, 9];
    let f = &|acc, x| i32::checked_add(2 * acc, x);
    let f_ref = &|acc, &x| i32::checked_add(2 * acc, x);
    assert_eq!(a.iter().cloned().try_fold(7, f), a.iter().try_fold(7, f_ref));
    assert_eq!(a.iter().cloned().try_rfold(7, f), a.iter().try_rfold(7, f_ref));

    let a = [10, 20, 30, 40, 100, 60, 70, 80, 90];
    let mut iter = a.iter().cloned();
    assert_eq!(iter.try_fold(0_i8, |acc, x| acc.checked_add(x)), None);
    assert_eq!(iter.next(), Some(60));
    let mut iter = a.iter().cloned();
    assert_eq!(iter.try_rfold(0_i8, |acc, x| acc.checked_add(x)), None);
    assert_eq!(iter.next_back(), Some(70));
}

#[test]
fn test_chain_try_folds() {
    let c = || (0..10).chain(10..20);

    let f = &|acc, x| i32::checked_add(2 * acc, x);
    assert_eq!(c().try_fold(7, f), (0..20).try_fold(7, f));
    assert_eq!(c().try_rfold(7, f), (0..20).rev().try_fold(7, f));

    let mut iter = c();
    assert_eq!(iter.position(|x| x == 5), Some(5));
    assert_eq!(iter.next(), Some(6), "stopped in front, state Both");
    assert_eq!(iter.position(|x| x == 13), Some(6));
    assert_eq!(iter.next(), Some(14), "stopped in back, state Back");
    assert_eq!(iter.try_fold(0, |acc, x| Some(acc + x)), Some((15..20).sum()));

    let mut iter = c().rev(); // use rev to access try_rfold
    assert_eq!(iter.position(|x| x == 15), Some(4));
    assert_eq!(iter.next(), Some(14), "stopped in back, state Both");
    assert_eq!(iter.position(|x| x == 5), Some(8));
    assert_eq!(iter.next(), Some(4), "stopped in front, state Front");
    assert_eq!(iter.try_fold(0, |acc, x| Some(acc + x)), Some((0..4).sum()));

    let mut iter = c();
    iter.by_ref().rev().nth(14); // skip the last 15, ending in state Front
    assert_eq!(iter.try_fold(7, f), (0..5).try_fold(7, f));

    let mut iter = c();
    iter.nth(14); // skip the first 15, ending in state Back
    assert_eq!(iter.try_rfold(7, f), (15..20).try_rfold(7, f));
}

#[test]
fn test_map_try_folds() {
    let f = &|acc, x| i32::checked_add(2 * acc, x);
    assert_eq!((0..10).map(|x| x + 3).try_fold(7, f), (3..13).try_fold(7, f));
    assert_eq!((0..10).map(|x| x + 3).try_rfold(7, f), (3..13).try_rfold(7, f));

    let mut iter = (0..40).map(|x| x + 10);
    assert_eq!(iter.try_fold(0, i8::checked_add), None);
    assert_eq!(iter.next(), Some(20));
    assert_eq!(iter.try_rfold(0, i8::checked_add), None);
    assert_eq!(iter.next_back(), Some(46));
}

#[test]
fn test_filter_try_folds() {
    fn p(&x: &i32) -> bool {
        0 <= x && x < 10
    }
    let f = &|acc, x| i32::checked_add(2 * acc, x);
    assert_eq!((-10..20).filter(p).try_fold(7, f), (0..10).try_fold(7, f));
    assert_eq!((-10..20).filter(p).try_rfold(7, f), (0..10).try_rfold(7, f));

    let mut iter = (0..40).filter(|&x| x % 2 == 1);
    assert_eq!(iter.try_fold(0, i8::checked_add), None);
    assert_eq!(iter.next(), Some(25));
    assert_eq!(iter.try_rfold(0, i8::checked_add), None);
    assert_eq!(iter.next_back(), Some(31));
}

#[test]
fn test_filter_map_try_folds() {
    let mp = &|x| if 0 <= x && x < 10 { Some(x * 2) } else { None };
    let f = &|acc, x| i32::checked_add(2 * acc, x);
    assert_eq!((-9..20).filter_map(mp).try_fold(7, f), (0..10).map(|x| 2 * x).try_fold(7, f));
    assert_eq!((-9..20).filter_map(mp).try_rfold(7, f), (0..10).map(|x| 2 * x).try_rfold(7, f));

    let mut iter = (0..40).filter_map(|x| if x % 2 == 1 { None } else { Some(x * 2 + 10) });
    assert_eq!(iter.try_fold(0, i8::checked_add), None);
    assert_eq!(iter.next(), Some(38));
    assert_eq!(iter.try_rfold(0, i8::checked_add), None);
    assert_eq!(iter.next_back(), Some(78));
}

#[test]
fn test_enumerate_try_folds() {
    let f = &|acc, (i, x)| usize::checked_add(2 * acc, x / (i + 1) + i);
    assert_eq!((9..18).enumerate().try_fold(7, f), (0..9).map(|i| (i, i + 9)).try_fold(7, f));
    assert_eq!((9..18).enumerate().try_rfold(7, f), (0..9).map(|i| (i, i + 9)).try_rfold(7, f));

    let mut iter = (100..200).enumerate();
    let f = &|acc, (i, x)| u8::checked_add(acc, u8::checked_div(x, i as u8 + 1)?);
    assert_eq!(iter.try_fold(0, f), None);
    assert_eq!(iter.next(), Some((7, 107)));
    assert_eq!(iter.try_rfold(0, f), None);
    assert_eq!(iter.next_back(), Some((11, 111)));
}

#[test]
fn test_peek_try_folds() {
    let f = &|acc, x| i32::checked_add(2 * acc, x);

    assert_eq!((1..20).peekable().try_fold(7, f), (1..20).try_fold(7, f));
    assert_eq!((1..20).peekable().try_rfold(7, f), (1..20).try_rfold(7, f));

    let mut iter = (1..20).peekable();
    assert_eq!(iter.peek(), Some(&1));
    assert_eq!(iter.try_fold(7, f), (1..20).try_fold(7, f));

    let mut iter = (1..20).peekable();
    assert_eq!(iter.peek(), Some(&1));
    assert_eq!(iter.try_rfold(7, f), (1..20).try_rfold(7, f));

    let mut iter = [100, 20, 30, 40, 50, 60, 70].iter().cloned().peekable();
    assert_eq!(iter.peek(), Some(&100));
    assert_eq!(iter.try_fold(0, i8::checked_add), None);
    assert_eq!(iter.peek(), Some(&40));

    let mut iter = [100, 20, 30, 40, 50, 60, 70].iter().cloned().peekable();
    assert_eq!(iter.peek(), Some(&100));
    assert_eq!(iter.try_rfold(0, i8::checked_add), None);
    assert_eq!(iter.peek(), Some(&100));
    assert_eq!(iter.next_back(), Some(50));

    let mut iter = (2..5).peekable();
    assert_eq!(iter.peek(), Some(&2));
    assert_eq!(iter.try_for_each(Err), Err(2));
    assert_eq!(iter.peek(), Some(&3));
    assert_eq!(iter.try_for_each(Err), Err(3));
    assert_eq!(iter.peek(), Some(&4));
    assert_eq!(iter.try_for_each(Err), Err(4));
    assert_eq!(iter.peek(), None);
    assert_eq!(iter.try_for_each(Err), Ok(()));

    let mut iter = (2..5).peekable();
    assert_eq!(iter.peek(), Some(&2));
    assert_eq!(iter.try_rfold((), |(), x| Err(x)), Err(4));
    assert_eq!(iter.peek(), Some(&2));
    assert_eq!(iter.try_rfold((), |(), x| Err(x)), Err(3));
    assert_eq!(iter.peek(), Some(&2));
    assert_eq!(iter.try_rfold((), |(), x| Err(x)), Err(2));
    assert_eq!(iter.peek(), None);
    assert_eq!(iter.try_rfold((), |(), x| Err(x)), Ok(()));
}

#[test]
fn test_skip_while_try_fold() {
    let f = &|acc, x| i32::checked_add(2 * acc, x);
    fn p(&x: &i32) -> bool {
        (x % 10) <= 5
    }
    assert_eq!((1..20).skip_while(p).try_fold(7, f), (6..20).try_fold(7, f));
    let mut iter = (1..20).skip_while(p);
    assert_eq!(iter.nth(5), Some(11));
    assert_eq!(iter.try_fold(7, f), (12..20).try_fold(7, f));

    let mut iter = (0..50).skip_while(|&x| (x % 20) < 15);
    assert_eq!(iter.try_fold(0, i8::checked_add), None);
    assert_eq!(iter.next(), Some(23));
}

#[test]
fn test_take_while_folds() {
    let f = &|acc, x| i32::checked_add(2 * acc, x);
    assert_eq!((1..20).take_while(|&x| x != 10).try_fold(7, f), (1..10).try_fold(7, f));
    let mut iter = (1..20).take_while(|&x| x != 10);
    assert_eq!(iter.try_fold(0, |x, y| Some(x + y)), Some((1..10).sum()));
    assert_eq!(iter.next(), None, "flag should be set");
    let iter = (1..20).take_while(|&x| x != 10);
    assert_eq!(iter.fold(0, |x, y| x + y), (1..10).sum());

    let mut iter = (10..50).take_while(|&x| x != 40);
    assert_eq!(iter.try_fold(0, i8::checked_add), None);
    assert_eq!(iter.next(), Some(20));
}

#[test]
fn test_skip_try_folds() {
    let f = &|acc, x| i32::checked_add(2 * acc, x);
    assert_eq!((1..20).skip(9).try_fold(7, f), (10..20).try_fold(7, f));
    assert_eq!((1..20).skip(9).try_rfold(7, f), (10..20).try_rfold(7, f));

    let mut iter = (0..30).skip(10);
    assert_eq!(iter.try_fold(0, i8::checked_add), None);
    assert_eq!(iter.next(), Some(20));
    assert_eq!(iter.try_rfold(0, i8::checked_add), None);
    assert_eq!(iter.next_back(), Some(24));
}

#[test]
fn test_skip_nth_back() {
    let xs = [0, 1, 2, 3, 4, 5];
    let mut it = xs.iter().skip(2);
    assert_eq!(it.nth_back(0), Some(&5));
    assert_eq!(it.nth_back(1), Some(&3));
    assert_eq!(it.nth_back(0), Some(&2));
    assert_eq!(it.nth_back(0), None);

    let ys = [2, 3, 4, 5];
    let mut ity = ys.iter();
    let mut it = xs.iter().skip(2);
    assert_eq!(it.nth_back(1), ity.nth_back(1));
    assert_eq!(it.clone().nth(0), ity.clone().nth(0));
    assert_eq!(it.nth_back(0), ity.nth_back(0));
    assert_eq!(it.clone().nth(0), ity.clone().nth(0));
    assert_eq!(it.nth_back(0), ity.nth_back(0));
    assert_eq!(it.clone().nth(0), ity.clone().nth(0));
    assert_eq!(it.nth_back(0), ity.nth_back(0));
    assert_eq!(it.clone().nth(0), ity.clone().nth(0));

    let mut it = xs.iter().skip(2);
    assert_eq!(it.nth_back(4), None);
    assert_eq!(it.nth_back(0), None);

    let mut it = xs.iter();
    it.by_ref().skip(2).nth_back(3);
    assert_eq!(it.next_back(), Some(&1));

    let mut it = xs.iter();
    it.by_ref().skip(2).nth_back(10);
    assert_eq!(it.next_back(), Some(&1));
}

#[test]
fn test_take_try_folds() {
    let f = &|acc, x| i32::checked_add(2 * acc, x);
    assert_eq!((10..30).take(10).try_fold(7, f), (10..20).try_fold(7, f));
    assert_eq!((10..30).take(10).try_rfold(7, f), (10..20).try_rfold(7, f));

    let mut iter = (10..30).take(20);
    assert_eq!(iter.try_fold(0, i8::checked_add), None);
    assert_eq!(iter.next(), Some(20));
    assert_eq!(iter.try_rfold(0, i8::checked_add), None);
    assert_eq!(iter.next_back(), Some(24));

    let mut iter = (2..20).take(3);
    assert_eq!(iter.try_for_each(Err), Err(2));
    assert_eq!(iter.try_for_each(Err), Err(3));
    assert_eq!(iter.try_for_each(Err), Err(4));
    assert_eq!(iter.try_for_each(Err), Ok(()));

    let mut iter = (2..20).take(3).rev();
    assert_eq!(iter.try_for_each(Err), Err(4));
    assert_eq!(iter.try_for_each(Err), Err(3));
    assert_eq!(iter.try_for_each(Err), Err(2));
    assert_eq!(iter.try_for_each(Err), Ok(()));
}

#[test]
fn test_flat_map_try_folds() {
    let f = &|acc, x| i32::checked_add(acc * 2 / 3, x);
    let mr = &|x| (5 * x)..(5 * x + 5);
    assert_eq!((0..10).flat_map(mr).try_fold(7, f), (0..50).try_fold(7, f));
    assert_eq!((0..10).flat_map(mr).try_rfold(7, f), (0..50).try_rfold(7, f));
    let mut iter = (0..10).flat_map(mr);
    iter.next();
    iter.next_back(); // have front and back iters in progress
    assert_eq!(iter.try_rfold(7, f), (1..49).try_rfold(7, f));

    let mut iter = (0..10).flat_map(|x| (4 * x)..(4 * x + 4));
    assert_eq!(iter.try_fold(0, i8::checked_add), None);
    assert_eq!(iter.next(), Some(17));
    assert_eq!(iter.try_rfold(0, i8::checked_add), None);
    assert_eq!(iter.next_back(), Some(35));
}

#[test]
fn test_flatten_try_folds() {
    let f = &|acc, x| i32::checked_add(acc * 2 / 3, x);
    let mr = &|x| (5 * x)..(5 * x + 5);
    assert_eq!((0..10).map(mr).flatten().try_fold(7, f), (0..50).try_fold(7, f));
    assert_eq!((0..10).map(mr).flatten().try_rfold(7, f), (0..50).try_rfold(7, f));
    let mut iter = (0..10).map(mr).flatten();
    iter.next();
    iter.next_back(); // have front and back iters in progress
    assert_eq!(iter.try_rfold(7, f), (1..49).try_rfold(7, f));

    let mut iter = (0..10).map(|x| (4 * x)..(4 * x + 4)).flatten();
    assert_eq!(iter.try_fold(0, i8::checked_add), None);
    assert_eq!(iter.next(), Some(17));
    assert_eq!(iter.try_rfold(0, i8::checked_add), None);
    assert_eq!(iter.next_back(), Some(35));
}

#[test]
fn test_functor_laws() {
    // identity:
    fn identity<T>(x: T) -> T {
        x
    }
    assert_eq!((0..10).map(identity).sum::<usize>(), (0..10).sum());

    // composition:
    fn f(x: usize) -> usize {
        x + 3
    }
    fn g(x: usize) -> usize {
        x * 2
    }
    fn h(x: usize) -> usize {
        g(f(x))
    }
    assert_eq!((0..10).map(f).map(g).sum::<usize>(), (0..10).map(h).sum());
}

#[test]
fn test_monad_laws_left_identity() {
    fn f(x: usize) -> impl Iterator<Item = usize> {
        (0..10).map(move |y| x * y)
    }
    assert_eq!(once(42).flat_map(f.clone()).sum::<usize>(), f(42).sum());
}

#[test]
fn test_monad_laws_right_identity() {
    assert_eq!((0..10).flat_map(|x| once(x)).sum::<usize>(), (0..10).sum());
}

#[test]
fn test_monad_laws_associativity() {
    fn f(x: usize) -> impl Iterator<Item = usize> {
        0..x
    }
    fn g(x: usize) -> impl Iterator<Item = usize> {
        (0..x).rev()
    }
    assert_eq!(
        (0..10).flat_map(f).flat_map(g).sum::<usize>(),
        (0..10).flat_map(|x| f(x).flat_map(g)).sum::<usize>()
    );
}

#[test]
fn test_is_sorted() {
    assert!([1, 2, 2, 9].iter().is_sorted());
    assert!(![1, 3, 2].iter().is_sorted());
    assert!([0].iter().is_sorted());
    assert!(std::iter::empty::<i32>().is_sorted());
    assert!(![0.0, 1.0, f32::NAN].iter().is_sorted());
    assert!([-2, -1, 0, 3].iter().is_sorted());
    assert!(![-2i32, -1, 0, 3].iter().is_sorted_by_key(|n| n.abs()));
    assert!(!["c", "bb", "aaa"].iter().is_sorted());
    assert!(["c", "bb", "aaa"].iter().is_sorted_by_key(|s| s.len()));
}

#[test]
fn test_partition() {
    fn check(xs: &mut [i32], ref p: impl Fn(&i32) -> bool, expected: usize) {
        let i = xs.iter_mut().partition_in_place(p);
        assert_eq!(expected, i);
        assert!(xs[..i].iter().all(p));
        assert!(!xs[i..].iter().any(p));
        assert!(xs.iter().is_partitioned(p));
        if i == 0 || i == xs.len() {
            assert!(xs.iter().rev().is_partitioned(p));
        } else {
            assert!(!xs.iter().rev().is_partitioned(p));
        }
    }

    check(&mut [], |_| true, 0);
    check(&mut [], |_| false, 0);

    check(&mut [0], |_| true, 1);
    check(&mut [0], |_| false, 0);

    check(&mut [-1, 1], |&x| x > 0, 1);
    check(&mut [-1, 1], |&x| x < 0, 1);

    let ref mut xs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
    check(xs, |_| true, 10);
    check(xs, |_| false, 0);
    check(xs, |&x| x % 2 == 0, 5); // evens
    check(xs, |&x| x % 2 == 1, 5); // odds
    check(xs, |&x| x % 3 == 0, 4); // multiple of 3
    check(xs, |&x| x % 4 == 0, 3); // multiple of 4
    check(xs, |&x| x % 5 == 0, 2); // multiple of 5
    check(xs, |&x| x < 3, 3); // small
    check(xs, |&x| x > 6, 3); // large
}

/// An iterator that panics whenever `next` or next_back` is called
/// after `None` has already been returned. This does not violate
/// `Iterator`'s contract. Used to test that iterator adaptors don't
/// poll their inner iterators after exhausting them.
struct NonFused<I> {
    iter: I,
    done: bool,
}

impl<I> NonFused<I> {
    fn new(iter: I) -> Self {
        Self { iter, done: false }
    }
}

impl<I> Iterator for NonFused<I>
where
    I: Iterator,
{
    type Item = I::Item;

    fn next(&mut self) -> Option<Self::Item> {
        assert!(!self.done, "this iterator has already returned None");
        self.iter.next().or_else(|| {
            self.done = true;
            None
        })
    }
}

impl<I> DoubleEndedIterator for NonFused<I>
where
    I: DoubleEndedIterator,
{
    fn next_back(&mut self) -> Option<Self::Item> {
        assert!(!self.done, "this iterator has already returned None");
        self.iter.next_back().or_else(|| {
            self.done = true;
            None
        })
    }
}

#[test]
fn test_peekable_non_fused() {
    let mut iter = NonFused::new(empty::<i32>()).peekable();

    assert_eq!(iter.peek(), None);
    assert_eq!(iter.next_back(), None);
}

#[test]
fn test_flatten_non_fused_outer() {
    let mut iter = NonFused::new(once(0..2)).flatten();

    assert_eq!(iter.next_back(), Some(1));
    assert_eq!(iter.next(), Some(0));
    assert_eq!(iter.next(), None);
}

#[test]
fn test_flatten_non_fused_inner() {
    let mut iter = once(0..1).chain(once(1..3)).flat_map(NonFused::new);

    assert_eq!(iter.next_back(), Some(2));
    assert_eq!(iter.next(), Some(0));
    assert_eq!(iter.next(), Some(1));
    assert_eq!(iter.next(), None);
}

#[test]
pub fn extend_for_unit() {
    let mut x = 0;
    {
        let iter = (0..5).map(|_| {
            x += 1;
        });
        ().extend(iter);
    }
    assert_eq!(x, 5);
}
