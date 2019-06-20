use core::cell::Cell;
use core::convert::TryFrom;
use core::iter::*;
use core::{i8, i16, isize};
use core::usize;

#[test]
fn test_lt() {
    let empty: [isize; 0] = [];
    let xs = [1,2,3];
    let ys = [1,2,0];

    assert!(!xs.iter().lt(ys.iter()));
    assert!(!xs.iter().le(ys.iter()));
    assert!( xs.iter().gt(ys.iter()));
    assert!( xs.iter().ge(ys.iter()));

    assert!( ys.iter().lt(xs.iter()));
    assert!( ys.iter().le(xs.iter()));
    assert!(!ys.iter().gt(xs.iter()));
    assert!(!ys.iter().ge(xs.iter()));

    assert!( empty.iter().lt(xs.iter()));
    assert!( empty.iter().le(xs.iter()));
    assert!(!empty.iter().gt(xs.iter()));
    assert!(!empty.iter().ge(xs.iter()));

    // Sequence with NaN
    let u = [1.0f64, 2.0];
    let v = [0.0f64/0.0, 3.0];

    assert!(!u.iter().lt(v.iter()));
    assert!(!u.iter().le(v.iter()));
    assert!(!u.iter().gt(v.iter()));
    assert!(!u.iter().ge(v.iter()));

    let a = [0.0f64/0.0];
    let b = [1.0f64];
    let c = [2.0f64];

    assert!(a.iter().lt(b.iter()) == (a[0] <  b[0]));
    assert!(a.iter().le(b.iter()) == (a[0] <= b[0]));
    assert!(a.iter().gt(b.iter()) == (a[0] >  b[0]));
    assert!(a.iter().ge(b.iter()) == (a[0] >= b[0]));

    assert!(c.iter().lt(b.iter()) == (c[0] <  b[0]));
    assert!(c.iter().le(b.iter()) == (c[0] <= b[0]));
    assert!(c.iter().gt(b.iter()) == (c[0] >  b[0]));
    assert!(c.iter().ge(b.iter()) == (c[0] >= b[0]));
}

#[test]
fn test_multi_iter() {
    let xs = [1,2,3,4];
    let ys = [4,3,2,1];
    assert!(xs.iter().eq(ys.iter().rev()));
    assert!(xs.iter().lt(xs.iter().skip(2)));
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
    let value = [1, 2, 3, 4, 5, 6].iter().cloned()
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
        fn next(&mut self) -> Option<Self::Item> { Some(21) }
        fn nth(&mut self, n: usize) -> Option<Self::Item> {
            self.0 += n as Bigger + 1;
            Some(42)
        }
    }

    let mut it = Test(0);
    let root = usize::MAX >> (::std::mem::size_of::<usize>() * 8 / 2);
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
    assert_eq!(it.size_hint(), (usize::MAX-1, None));

    // still infinite with larger step
    let mut it = StubSizeHint(7, None).step_by(3);
    assert_eq!(it.size_hint(), (3, None));
    it.next();
    assert_eq!(it.size_hint(), (2, None));

    // propagates ExactSizeIterator
    let a = [1,2,3,4,5];
    let it = a.iter().step_by(2);
    assert_eq!(it.len(), 3);

    // Cannot be TrustedLen as a step greater than one makes an iterator
    // with (usize::MAX, None) no longer meet the safety requirements
    trait TrustedLenCheck { fn test(self) -> bool; }
    impl<T:Iterator> TrustedLenCheck for T {
        default fn test(self) -> bool { false }
    }
    impl<T:TrustedLen> TrustedLenCheck for T {
        fn test(self) -> bool { true }
    }
    assert!(TrustedLenCheck::test(a.iter()));
    assert!(!TrustedLenCheck::test(a.iter().step_by(1)));
}

#[test]
fn test_filter_map() {
    let it = (0..).step_by(1).take(10)
        .filter_map(|x| if x % 2 == 0 { Some(x*x) } else { None });
    assert_eq!(it.collect::<Vec<usize>>(), [0*0, 2*2, 4*4, 6*6, 8*8]);
}

#[test]
fn test_filter_map_fold() {
    let xs = [0, 1, 2, 3, 4, 5, 6, 7, 8];
    let ys = [0*0, 2*2, 4*4, 6*6, 8*8];
    let it = xs.iter().filter_map(|&x| if x % 2 == 0 { Some(x*x) } else { None });
    let i = it.fold(0, |i, x| {
        assert_eq!(x, ys[i]);
        i + 1
    });
    assert_eq!(i, ys.len());

    let it = xs.iter().filter_map(|&x| if x % 2 == 0 { Some(x*x) } else { None });
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

/// This is an iterator that follows the Iterator contract,
/// but it is not fused. After having returned None once, it will start
/// producing elements if .next() is called again.
pub struct CycleIter<'a, T> {
    index: usize,
    data: &'a [T],
}

pub fn cycle<T>(data: &[T]) -> CycleIter<'_, T> {
    CycleIter {
        index: 0,
        data,
    }
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
        if n > data.len() { break; }
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
        assert_eq!(it.len(), xs.len()-5-i);
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
    let mut it = xs.iter().take(5);
    let mut i = 0;
    assert_eq!(it.len(), 5);
    while let Some(&x) = it.next() {
        assert_eq!(x, ys[i]);
        i += 1;
        assert_eq!(it.len(), 5-i);
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
fn test_iterator_take_short() {
    let xs = [0, 1, 2, 3];
    let ys = [0, 1, 2, 3];
    let mut it = xs.iter().take(5);
    let mut i = 0;
    assert_eq!(it.len(), 4);
    while let Some(&x) = it.next() {
        assert_eq!(x, ys[i]);
        i += 1;
        assert_eq!(it.len(), 4-i);
    }
    assert_eq!(i, ys.len());
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
    let mut it = xs.iter().flat_map(|&x| x..x+3);
    assert_eq!(it.next(), Some(0));
    assert_eq!(it.next_back(), Some(8));
    let i = it.fold(0, |i, x| {
        assert_eq!(x, ys[i]);
        i + 1
    });
    assert_eq!(i, ys.len());

    let mut it = xs.iter().flat_map(|&x| x..x+3);
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
    let mut it = xs.iter().map(|&x| x..x+3).flatten();
    assert_eq!(it.next(), Some(0));
    assert_eq!(it.next_back(), Some(8));
    let i = it.fold(0, |i, x| {
        assert_eq!(x, ys[i]);
        i + 1
    });
    assert_eq!(i, ys.len());

    let mut it = xs.iter().map(|&x| x..x+3).flatten();
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

    let ys = xs.iter()
               .cloned()
               .inspect(|_| n += 1)
               .collect::<Vec<usize>>();

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
    assert_eq!(c.clone().skip_while(|_| false).size_hint(), (0, None));
    assert_eq!(c.clone().enumerate().size_hint(), (usize::MAX, None));
    assert_eq!(c.clone().chain(vi.clone().cloned()).size_hint(), (usize::MAX, None));
    assert_eq!(c.clone().zip(vi.clone()).size_hint(), (10, Some(10)));
    assert_eq!(c.clone().scan(0, |_,_| Some(0)).size_hint(), (0, None));
    assert_eq!(c.clone().filter(|_| false).size_hint(), (0, None));
    assert_eq!(c.clone().map(|_| 0).size_hint(), (usize::MAX, None));
    assert_eq!(c.filter_map(|_| Some(0)).size_hint(), (0, None));

    assert_eq!(vi.clone().take(5).size_hint(), (5, Some(5)));
    assert_eq!(vi.clone().take(12).size_hint(), (10, Some(10)));
    assert_eq!(vi.clone().skip(3).size_hint(), (7, Some(7)));
    assert_eq!(vi.clone().skip(12).size_hint(), (0, Some(0)));
    assert_eq!(vi.clone().take_while(|_| false).size_hint(), (0, Some(10)));
    assert_eq!(vi.clone().skip_while(|_| false).size_hint(), (0, Some(10)));
    assert_eq!(vi.clone().enumerate().size_hint(), (10, Some(10)));
    assert_eq!(vi.clone().chain(v2).size_hint(), (13, Some(13)));
    assert_eq!(vi.clone().zip(v2).size_hint(), (3, Some(3)));
    assert_eq!(vi.clone().scan(0, |_,_| Some(0)).size_hint(), (0, Some(10)));
    assert_eq!(vi.clone().filter(|_| false).size_hint(), (0, Some(10)));
    assert_eq!(vi.clone().map(|&i| i+1).size_hint(), (10, Some(10)));
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
        if x % 2 == 0 {
            Some(x / 2)
        } else {
            None
        }
    }
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
    assert!(it.rev().cloned().collect::<Vec<isize>>() ==
            vec![16, 14, 12, 10, 8, 6]);
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
    impl CrazyIterator { fn new() -> CrazyIterator { CrazyIterator(false) } }
    impl Iterator for CrazyIterator {
        type Item = i32;
        fn next(&mut self) -> Option<i32> {
            if self.0 { Some(99) } else { self.0 = true; None }
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
    fn f(xy: &(isize, char)) -> bool { let (_x, y) = *xy; y == 'b' }
    fn g(xy: &(isize, char)) -> bool { let (_x, y) = *xy; y == 'd' }
    let v = [(0, 'a'), (1, 'b'), (2, 'c'), (3, 'b')];

    assert_eq!(v.iter().rposition(f), Some(3));
    assert!(v.iter().rposition(g).is_none());
}

#[test]
#[should_panic]
fn test_rposition_panic() {
    let v: [(Box<_>, Box<_>); 4] =
        [(box 0, box 0), (box 0, box 0),
         (box 0, box 0), (box 0, box 0)];
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
    let u = [0,1];
    let v = [5,6,7,8];
    let mut it = u.iter().flat_map(|x| &v[*x..v.len()]);
    assert_eq!(it.next_back().unwrap(), &8);
    assert_eq!(it.next().unwrap(),      &5);
    assert_eq!(it.next_back().unwrap(), &7);
    assert_eq!(it.next_back().unwrap(), &6);
    assert_eq!(it.next_back().unwrap(), &8);
    assert_eq!(it.next().unwrap(),      &6);
    assert_eq!(it.next_back().unwrap(), &7);
    assert_eq!(it.next_back(), None);
    assert_eq!(it.next(),      None);
    assert_eq!(it.next_back(), None);
}

#[test]
fn test_double_ended_flatten() {
    let u = [0,1];
    let v = [5,6,7,8];
    let mut it = u.iter().map(|x| &v[*x..v.len()]).flatten();
    assert_eq!(it.next_back().unwrap(), &8);
    assert_eq!(it.next().unwrap(),      &5);
    assert_eq!(it.next_back().unwrap(), &7);
    assert_eq!(it.next_back().unwrap(), &6);
    assert_eq!(it.next_back().unwrap(), &8);
    assert_eq!(it.next().unwrap(),      &6);
    assert_eq!(it.next_back().unwrap(), &7);
    assert_eq!(it.next_back(), None);
    assert_eq!(it.next(),      None);
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
    assert_eq!((-2..isize::MAX).size_hint(),
               (isize::MAX as usize + 2, Some(isize::MAX as usize + 2)));
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

    let mut r = 10..=10;
    assert_eq!(r.next_back(), Some(10));
    assert!(r.is_empty());
    assert_eq!(r.next_back(), None);

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

    let mut r = 10_u8..=20;
    assert_eq!(r.nth(2), Some(12));
    assert_eq!(r, 13..=20);
    assert_eq!(r.nth(2), Some(15));
    assert_eq!(r, 16..=20);
    assert_eq!(r.is_empty(), false);
    assert_eq!(ExactSizeIterator::is_empty(&r), false);
    assert_eq!(r.nth(10), None);
    assert_eq!(r.is_empty(), true);
    assert_eq!(ExactSizeIterator::is_empty(&r), true);
}

#[test]
fn test_range_inclusive_nth_back() {
    assert_eq!((10..=15).nth_back(0), Some(15));
    assert_eq!((10..=15).nth_back(1), Some(14));
    assert_eq!((10..=15).nth_back(5), Some(10));
    assert_eq!((10..=15).nth_back(6), None);
    assert_eq!((-120..=80_i8).nth_back(200), Some(-120));

    let mut r = 10_u8..=20;
    assert_eq!(r.nth_back(2), Some(18));
    assert_eq!(r, 10..=17);
    assert_eq!(r.nth_back(2), Some(15));
    assert_eq!(r, 10..=14);
    assert_eq!(r.is_empty(), false);
    assert_eq!(ExactSizeIterator::is_empty(&r), false);
    assert_eq!(r.nth_back(10), None);
    assert_eq!(r.is_empty(), true);
    assert_eq!(ExactSizeIterator::is_empty(&r), true);
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
    assert_eq!(it.try_fold(0, |a,b| Some(a+b)), Some(165));
    assert!(it.is_empty());
    assert_eq!(it.try_fold(0, |a,b| Some(a+b)), Some(0));
    assert!(it.is_empty());

    let mut it = 10..=20;
    assert_eq!(it.try_rfold(0, |a,b| Some(a+b)), Some(165));
    assert!(it.is_empty());
    assert_eq!(it.try_rfold(0, |a,b| Some(a+b)), Some(0));
    assert!(it.is_empty());
}

#[test]
fn test_range_size_hint() {
    use core::usize::MAX as UMAX;
    assert_eq!((0..0usize).size_hint(), (0, Some(0)));
    assert_eq!((0..100usize).size_hint(), (100, Some(100)));
    assert_eq!((0..UMAX).size_hint(), (UMAX, Some(UMAX)));

    let umax = u128::try_from(UMAX).unwrap();
    assert_eq!((0..0u128).size_hint(), (0, Some(0)));
    assert_eq!((0..100u128).size_hint(), (100, Some(100)));
    assert_eq!((0..umax).size_hint(), (UMAX, Some(UMAX)));
    assert_eq!((0..umax + 1).size_hint(), (UMAX, None));

    use core::isize::{MAX as IMAX, MIN as IMIN};
    assert_eq!((0..0isize).size_hint(), (0, Some(0)));
    assert_eq!((-100..100isize).size_hint(), (200, Some(200)));
    assert_eq!((IMIN..IMAX).size_hint(), (UMAX, Some(UMAX)));

    let imin = i128::try_from(IMIN).unwrap();
    let imax = i128::try_from(IMAX).unwrap();
    assert_eq!((0..0i128).size_hint(), (0, Some(0)));
    assert_eq!((-100..100i128).size_hint(), (200, Some(200)));
    assert_eq!((imin..imax).size_hint(), (UMAX, Some(UMAX)));
    assert_eq!((imin..imax + 1).size_hint(), (UMAX, None));
}

#[test]
fn test_range_inclusive_size_hint() {
    use core::usize::MAX as UMAX;
    assert_eq!((1..=0usize).size_hint(), (0, Some(0)));
    assert_eq!((0..=0usize).size_hint(), (1, Some(1)));
    assert_eq!((0..=100usize).size_hint(), (101, Some(101)));
    assert_eq!((0..=UMAX - 1).size_hint(), (UMAX, Some(UMAX)));
    assert_eq!((0..=UMAX).size_hint(), (UMAX, None));

    let umax = u128::try_from(UMAX).unwrap();
    assert_eq!((1..=0u128).size_hint(), (0, Some(0)));
    assert_eq!((0..=0u128).size_hint(), (1, Some(1)));
    assert_eq!((0..=100u128).size_hint(), (101, Some(101)));
    assert_eq!((0..=umax - 1).size_hint(), (UMAX, Some(UMAX)));
    assert_eq!((0..=umax).size_hint(), (UMAX, None));
    assert_eq!((0..=umax + 1).size_hint(), (UMAX, None));

    use core::isize::{MAX as IMAX, MIN as IMIN};
    assert_eq!((0..=-1isize).size_hint(), (0, Some(0)));
    assert_eq!((0..=0isize).size_hint(), (1, Some(1)));
    assert_eq!((-100..=100isize).size_hint(), (201, Some(201)));
    assert_eq!((IMIN..=IMAX - 1).size_hint(), (UMAX, Some(UMAX)));
    assert_eq!((IMIN..=IMAX).size_hint(), (UMAX, None));

    let imin = i128::try_from(IMIN).unwrap();
    let imax = i128::try_from(IMAX).unwrap();
    assert_eq!((0..=-1i128).size_hint(), (0, Some(0)));
    assert_eq!((0..=0i128).size_hint(), (1, Some(1)));
    assert_eq!((-100..=100i128).size_hint(), (201, Some(201)));
    assert_eq!((imin..=imax - 1).size_hint(), (UMAX, Some(UMAX)));
    assert_eq!((imin..=imax).size_hint(), (UMAX, None));
    assert_eq!((imin..=imax + 1).size_hint(), (UMAX, None));
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
    assert_eq!(repeat_with(|| 42).take(usize::MAX).size_hint(),
               (usize::MAX, Some(usize::MAX)));
}

#[test]
fn test_repeat_with_take_collect() {
    let mut curr = 1;
    let v: Vec<_> = repeat_with(|| { let tmp = curr; curr *= 2; tmp })
                      .take(5).collect();
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
fn test_step_replace_unsigned() {
    let mut x = 4u32;
    let y = x.replace_zero();
    assert_eq!(x, 0);
    assert_eq!(y, 4);

    x = 5;
    let y = x.replace_one();
    assert_eq!(x, 1);
    assert_eq!(y, 5);
}

#[test]
fn test_step_replace_signed() {
    let mut x = 4i32;
    let y = x.replace_zero();
    assert_eq!(x, 0);
    assert_eq!(y, 4);

    x = 5;
    let y = x.replace_one();
    assert_eq!(x, 1);
    assert_eq!(y, 5);
}

#[test]
fn test_step_replace_no_between() {
    let mut x = 4u128;
    let y = x.replace_zero();
    assert_eq!(x, 0);
    assert_eq!(y, 4);

    x = 5;
    let y = x.replace_one();
    assert_eq!(x, 1);
    assert_eq!(y, 5);
}

#[test]
fn test_rev_try_folds() {
    let f = &|acc, x| i32::checked_add(2*acc, x);
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
    let f = &|acc, x| i32::checked_add(2*acc, x);
    let f_ref = &|acc, &x| i32::checked_add(2*acc, x);
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

    let f = &|acc, x| i32::checked_add(2*acc, x);
    assert_eq!(c().try_fold(7, f), (0..20).try_fold(7, f));
    assert_eq!(c().try_rfold(7, f), (0..20).rev().try_fold(7, f));

    let mut iter = c();
    assert_eq!(iter.position(|x| x == 5), Some(5));
    assert_eq!(iter.next(), Some(6), "stopped in front, state Both");
    assert_eq!(iter.position(|x| x == 13), Some(6));
    assert_eq!(iter.next(), Some(14), "stopped in back, state Back");
    assert_eq!(iter.try_fold(0, |acc, x| Some(acc+x)), Some((15..20).sum()));

    let mut iter = c().rev(); // use rev to access try_rfold
    assert_eq!(iter.position(|x| x == 15), Some(4));
    assert_eq!(iter.next(), Some(14), "stopped in back, state Both");
    assert_eq!(iter.position(|x| x == 5), Some(8));
    assert_eq!(iter.next(), Some(4), "stopped in front, state Front");
    assert_eq!(iter.try_fold(0, |acc, x| Some(acc+x)), Some((0..4).sum()));

    let mut iter = c();
    iter.by_ref().rev().nth(14); // skip the last 15, ending in state Front
    assert_eq!(iter.try_fold(7, f), (0..5).try_fold(7, f));

    let mut iter = c();
    iter.nth(14); // skip the first 15, ending in state Back
    assert_eq!(iter.try_rfold(7, f), (15..20).try_rfold(7, f));
}

#[test]
fn test_map_try_folds() {
    let f = &|acc, x| i32::checked_add(2*acc, x);
    assert_eq!((0..10).map(|x| x+3).try_fold(7, f), (3..13).try_fold(7, f));
    assert_eq!((0..10).map(|x| x+3).try_rfold(7, f), (3..13).try_rfold(7, f));

    let mut iter = (0..40).map(|x| x+10);
    assert_eq!(iter.try_fold(0, i8::checked_add), None);
    assert_eq!(iter.next(), Some(20));
    assert_eq!(iter.try_rfold(0, i8::checked_add), None);
    assert_eq!(iter.next_back(), Some(46));
}

#[test]
fn test_filter_try_folds() {
    fn p(&x: &i32) -> bool { 0 <= x && x < 10 }
    let f = &|acc, x| i32::checked_add(2*acc, x);
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
    let mp = &|x| if 0 <= x && x < 10 { Some(x*2) } else { None };
    let f = &|acc, x| i32::checked_add(2*acc, x);
    assert_eq!((-9..20).filter_map(mp).try_fold(7, f), (0..10).map(|x| 2*x).try_fold(7, f));
    assert_eq!((-9..20).filter_map(mp).try_rfold(7, f), (0..10).map(|x| 2*x).try_rfold(7, f));

    let mut iter = (0..40).filter_map(|x| if x%2 == 1 { None } else { Some(x*2 + 10) });
    assert_eq!(iter.try_fold(0, i8::checked_add), None);
    assert_eq!(iter.next(), Some(38));
    assert_eq!(iter.try_rfold(0, i8::checked_add), None);
    assert_eq!(iter.next_back(), Some(78));
}

#[test]
fn test_enumerate_try_folds() {
    let f = &|acc, (i, x)| usize::checked_add(2*acc, x/(i+1) + i);
    assert_eq!((9..18).enumerate().try_fold(7, f), (0..9).map(|i| (i, i+9)).try_fold(7, f));
    assert_eq!((9..18).enumerate().try_rfold(7, f), (0..9).map(|i| (i, i+9)).try_rfold(7, f));

    let mut iter = (100..200).enumerate();
    let f = &|acc, (i, x)| u8::checked_add(acc, u8::checked_div(x, i as u8 + 1)?);
    assert_eq!(iter.try_fold(0, f), None);
    assert_eq!(iter.next(), Some((7, 107)));
    assert_eq!(iter.try_rfold(0, f), None);
    assert_eq!(iter.next_back(), Some((11, 111)));
}

#[test]
fn test_peek_try_fold() {
    let f = &|acc, x| i32::checked_add(2*acc, x);
    assert_eq!((1..20).peekable().try_fold(7, f), (1..20).try_fold(7, f));
    let mut iter = (1..20).peekable();
    assert_eq!(iter.peek(), Some(&1));
    assert_eq!(iter.try_fold(7, f), (1..20).try_fold(7, f));

    let mut iter = [100, 20, 30, 40, 50, 60, 70].iter().cloned().peekable();
    assert_eq!(iter.peek(), Some(&100));
    assert_eq!(iter.try_fold(0, i8::checked_add), None);
    assert_eq!(iter.peek(), Some(&40));
}

#[test]
fn test_skip_while_try_fold() {
    let f = &|acc, x| i32::checked_add(2*acc, x);
    fn p(&x: &i32) -> bool { (x % 10) <= 5 }
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
    let f = &|acc, x| i32::checked_add(2*acc, x);
    assert_eq!((1..20).take_while(|&x| x != 10).try_fold(7, f), (1..10).try_fold(7, f));
    let mut iter = (1..20).take_while(|&x| x != 10);
    assert_eq!(iter.try_fold(0, |x, y| Some(x+y)), Some((1..10).sum()));
    assert_eq!(iter.next(), None, "flag should be set");
    let iter = (1..20).take_while(|&x| x != 10);
    assert_eq!(iter.fold(0, |x, y| x+y), (1..10).sum());

    let mut iter = (10..50).take_while(|&x| x != 40);
    assert_eq!(iter.try_fold(0, i8::checked_add), None);
    assert_eq!(iter.next(), Some(20));
}

#[test]
fn test_skip_try_folds() {
    let f = &|acc, x| i32::checked_add(2*acc, x);
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
    let f = &|acc, x| i32::checked_add(2*acc, x);
    assert_eq!((10..30).take(10).try_fold(7, f), (10..20).try_fold(7, f));
    //assert_eq!((10..30).take(10).try_rfold(7, f), (10..20).try_rfold(7, f));

    let mut iter = (10..30).take(20);
    assert_eq!(iter.try_fold(0, i8::checked_add), None);
    assert_eq!(iter.next(), Some(20));
    //assert_eq!(iter.try_rfold(0, i8::checked_add), None);
    //assert_eq!(iter.next_back(), Some(24));
}

#[test]
fn test_flat_map_try_folds() {
    let f = &|acc, x| i32::checked_add(acc*2/3, x);
    let mr = &|x| (5*x)..(5*x + 5);
    assert_eq!((0..10).flat_map(mr).try_fold(7, f), (0..50).try_fold(7, f));
    assert_eq!((0..10).flat_map(mr).try_rfold(7, f), (0..50).try_rfold(7, f));
    let mut iter = (0..10).flat_map(mr);
    iter.next(); iter.next_back(); // have front and back iters in progress
    assert_eq!(iter.try_rfold(7, f), (1..49).try_rfold(7, f));

    let mut iter = (0..10).flat_map(|x| (4*x)..(4*x + 4));
    assert_eq!(iter.try_fold(0, i8::checked_add), None);
    assert_eq!(iter.next(), Some(17));
    assert_eq!(iter.try_rfold(0, i8::checked_add), None);
    assert_eq!(iter.next_back(), Some(35));
}

#[test]
fn test_flatten_try_folds() {
    let f = &|acc, x| i32::checked_add(acc*2/3, x);
    let mr = &|x| (5*x)..(5*x + 5);
    assert_eq!((0..10).map(mr).flatten().try_fold(7, f), (0..50).try_fold(7, f));
    assert_eq!((0..10).map(mr).flatten().try_rfold(7, f), (0..50).try_rfold(7, f));
    let mut iter = (0..10).map(mr).flatten();
    iter.next(); iter.next_back(); // have front and back iters in progress
    assert_eq!(iter.try_rfold(7, f), (1..49).try_rfold(7, f));

    let mut iter = (0..10).map(|x| (4*x)..(4*x + 4)).flatten();
    assert_eq!(iter.try_fold(0, i8::checked_add), None);
    assert_eq!(iter.next(), Some(17));
    assert_eq!(iter.try_rfold(0, i8::checked_add), None);
    assert_eq!(iter.next_back(), Some(35));
}

#[test]
fn test_functor_laws() {
    // identity:
    fn identity<T>(x: T) -> T { x }
    assert_eq!((0..10).map(identity).sum::<usize>(), (0..10).sum());

    // composition:
    fn f(x: usize) -> usize { x + 3 }
    fn g(x: usize) -> usize { x * 2 }
    fn h(x: usize) -> usize { g(f(x)) }
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
    fn f(x: usize) -> impl Iterator<Item = usize> { 0..x }
    fn g(x: usize) -> impl Iterator<Item = usize> { (0..x).rev() }
    assert_eq!((0..10).flat_map(f).flat_map(g).sum::<usize>(),
                (0..10).flat_map(|x| f(x).flat_map(g)).sum::<usize>());
}

#[test]
fn test_is_sorted() {
    assert!([1, 2, 2, 9].iter().is_sorted());
    assert!(![1, 3, 2].iter().is_sorted());
    assert!([0].iter().is_sorted());
    assert!(std::iter::empty::<i32>().is_sorted());
    assert!(![0.0, 1.0, std::f32::NAN].iter().is_sorted());
    assert!([-2, -1, 0, 3].iter().is_sorted());
    assert!(![-2i32, -1, 0, 3].iter().is_sorted_by_key(|n| n.abs()));
    assert!(!["c", "bb", "aaa"].iter().is_sorted());
    assert!(["c", "bb", "aaa"].iter().is_sorted_by_key(|s| s.len()));
}
