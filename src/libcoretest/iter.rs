// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use core::iter::*;
use core::iter::order::*;
use core::uint;
use core::cmp;
use core::num;

use test::Bencher;

#[test]
fn test_lt() {
    let empty: [int, ..0] = [];
    let xs = [1i,2,3];
    let ys = [1i,2,0];

    assert!(!lt(xs.iter(), ys.iter()));
    assert!(!le(xs.iter(), ys.iter()));
    assert!( gt(xs.iter(), ys.iter()));
    assert!( ge(xs.iter(), ys.iter()));

    assert!( lt(ys.iter(), xs.iter()));
    assert!( le(ys.iter(), xs.iter()));
    assert!(!gt(ys.iter(), xs.iter()));
    assert!(!ge(ys.iter(), xs.iter()));

    assert!( lt(empty.iter(), xs.iter()));
    assert!( le(empty.iter(), xs.iter()));
    assert!(!gt(empty.iter(), xs.iter()));
    assert!(!ge(empty.iter(), xs.iter()));

    // Sequence with NaN
    let u = [1.0f64, 2.0];
    let v = [0.0f64/0.0, 3.0];

    assert!(!lt(u.iter(), v.iter()));
    assert!(!le(u.iter(), v.iter()));
    assert!(!gt(u.iter(), v.iter()));
    assert!(!ge(u.iter(), v.iter()));

    let a = [0.0f64/0.0];
    let b = [1.0f64];
    let c = [2.0f64];

    assert!(lt(a.iter(), b.iter()) == (a[0] <  b[0]));
    assert!(le(a.iter(), b.iter()) == (a[0] <= b[0]));
    assert!(gt(a.iter(), b.iter()) == (a[0] >  b[0]));
    assert!(ge(a.iter(), b.iter()) == (a[0] >= b[0]));

    assert!(lt(c.iter(), b.iter()) == (c[0] <  b[0]));
    assert!(le(c.iter(), b.iter()) == (c[0] <= b[0]));
    assert!(gt(c.iter(), b.iter()) == (c[0] >  b[0]));
    assert!(ge(c.iter(), b.iter()) == (c[0] >= b[0]));
}

#[test]
fn test_multi_iter() {
    let xs = [1i,2,3,4];
    let ys = [4i,3,2,1];
    assert!(eq(xs.iter(), ys.iter().rev()));
    assert!(lt(xs.iter(), xs.iter().skip(2)));
}

#[test]
fn test_counter_from_iter() {
    let it = count(0i, 5).take(10);
    let xs: Vec<int> = FromIterator::from_iter(it);
    assert!(xs == vec![0, 5, 10, 15, 20, 25, 30, 35, 40, 45]);
}

#[test]
fn test_iterator_chain() {
    let xs = [0u, 1, 2, 3, 4, 5];
    let ys = [30u, 40, 50, 60];
    let expected = [0, 1, 2, 3, 4, 5, 30, 40, 50, 60];
    let mut it = xs.iter().chain(ys.iter());
    let mut i = 0;
    for &x in it {
        assert_eq!(x, expected[i]);
        i += 1;
    }
    assert_eq!(i, expected.len());

    let ys = count(30u, 10).take(4);
    let mut it = xs.iter().map(|&x| x).chain(ys);
    let mut i = 0;
    for x in it {
        assert_eq!(x, expected[i]);
        i += 1;
    }
    assert_eq!(i, expected.len());
}

#[test]
fn test_filter_map() {
    let mut it = count(0u, 1u).take(10)
        .filter_map(|x| if x % 2 == 0 { Some(x*x) } else { None });
    assert!(it.collect::<Vec<uint>>() == vec![0*0, 2*2, 4*4, 6*6, 8*8]);
}

#[test]
fn test_iterator_enumerate() {
    let xs = [0u, 1, 2, 3, 4, 5];
    let mut it = xs.iter().enumerate();
    for (i, &x) in it {
        assert_eq!(i, x);
    }
}

#[test]
fn test_iterator_peekable() {
    let xs = vec![0u, 1, 2, 3, 4, 5];
    let mut it = xs.iter().map(|&x|x).peekable();
    assert_eq!(it.peek().unwrap(), &0);
    assert_eq!(it.next().unwrap(), 0);
    assert_eq!(it.next().unwrap(), 1);
    assert_eq!(it.next().unwrap(), 2);
    assert_eq!(it.peek().unwrap(), &3);
    assert_eq!(it.peek().unwrap(), &3);
    assert_eq!(it.next().unwrap(), 3);
    assert_eq!(it.next().unwrap(), 4);
    assert_eq!(it.peek().unwrap(), &5);
    assert_eq!(it.next().unwrap(), 5);
    assert!(it.peek().is_none());
    assert!(it.next().is_none());
}

#[test]
fn test_iterator_take_while() {
    let xs = [0u, 1, 2, 3, 5, 13, 15, 16, 17, 19];
    let ys = [0u, 1, 2, 3, 5, 13];
    let mut it = xs.iter().take_while(|&x| *x < 15u);
    let mut i = 0;
    for x in it {
        assert_eq!(*x, ys[i]);
        i += 1;
    }
    assert_eq!(i, ys.len());
}

#[test]
fn test_iterator_skip_while() {
    let xs = [0u, 1, 2, 3, 5, 13, 15, 16, 17, 19];
    let ys = [15, 16, 17, 19];
    let mut it = xs.iter().skip_while(|&x| *x < 15u);
    let mut i = 0;
    for x in it {
        assert_eq!(*x, ys[i]);
        i += 1;
    }
    assert_eq!(i, ys.len());
}

#[test]
fn test_iterator_skip() {
    let xs = [0u, 1, 2, 3, 5, 13, 15, 16, 17, 19, 20, 30];
    let ys = [13, 15, 16, 17, 19, 20, 30];
    let mut it = xs.iter().skip(5);
    let mut i = 0;
    for &x in it {
        assert_eq!(x, ys[i]);
        i += 1;
    }
    assert_eq!(i, ys.len());
}

#[test]
fn test_iterator_take() {
    let xs = [0u, 1, 2, 3, 5, 13, 15, 16, 17, 19];
    let ys = [0u, 1, 2, 3, 5];
    let mut it = xs.iter().take(5);
    let mut i = 0;
    for &x in it {
        assert_eq!(x, ys[i]);
        i += 1;
    }
    assert_eq!(i, ys.len());
}

#[test]
fn test_iterator_scan() {
    // test the type inference
    fn add(old: &mut int, new: &uint) -> Option<f64> {
        *old += *new as int;
        Some(*old as f64)
    }
    let xs = [0u, 1, 2, 3, 4];
    let ys = [0f64, 1.0, 3.0, 6.0, 10.0];

    let mut it = xs.iter().scan(0, add);
    let mut i = 0;
    for x in it {
        assert_eq!(x, ys[i]);
        i += 1;
    }
    assert_eq!(i, ys.len());
}

#[test]
fn test_iterator_flat_map() {
    let xs = [0u, 3, 6];
    let ys = [0u, 1, 2, 3, 4, 5, 6, 7, 8];
    let mut it = xs.iter().flat_map(|&x| count(x, 1).take(3));
    let mut i = 0;
    for x in it {
        assert_eq!(x, ys[i]);
        i += 1;
    }
    assert_eq!(i, ys.len());
}

#[test]
fn test_inspect() {
    let xs = [1u, 2, 3, 4];
    let mut n = 0;

    let ys = xs.iter()
               .map(|&x| x)
               .inspect(|_| n += 1)
               .collect::<Vec<uint>>();

    assert_eq!(n, xs.len());
    assert_eq!(xs.as_slice(), ys.as_slice());
}

#[test]
fn test_unfoldr() {
    fn count(st: &mut uint) -> Option<uint> {
        if *st < 10 {
            let ret = Some(*st);
            *st += 1;
            ret
        } else {
            None
        }
    }

    let mut it = Unfold::new(0, count);
    let mut i = 0;
    for counted in it {
        assert_eq!(counted, i);
        i += 1;
    }
    assert_eq!(i, 10);
}

#[test]
fn test_cycle() {
    let cycle_len = 3;
    let it = count(0u, 1).take(cycle_len).cycle();
    assert_eq!(it.size_hint(), (uint::MAX, None));
    for (i, x) in it.take(100).enumerate() {
        assert_eq!(i % cycle_len, x);
    }

    let mut it = count(0u, 1).take(0).cycle();
    assert_eq!(it.size_hint(), (0, Some(0)));
    assert_eq!(it.next(), None);
}

#[test]
fn test_iterator_nth() {
    let v = &[0i, 1, 2, 3, 4];
    for i in range(0u, v.len()) {
        assert_eq!(v.iter().nth(i).unwrap(), &v[i]);
    }
    assert_eq!(v.iter().nth(v.len()), None);
}

#[test]
fn test_iterator_last() {
    let v = &[0i, 1, 2, 3, 4];
    assert_eq!(v.iter().last().unwrap(), &4);
    assert_eq!(v.slice(0, 1).iter().last().unwrap(), &0);
}

#[test]
fn test_iterator_len() {
    let v = &[0i, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    assert_eq!(v.slice(0, 4).iter().count(), 4);
    assert_eq!(v.slice(0, 10).iter().count(), 10);
    assert_eq!(v.slice(0, 0).iter().count(), 0);
}

#[test]
fn test_iterator_sum() {
    let v = &[0i, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    assert_eq!(v.slice(0, 4).iter().map(|&x| x).sum(), 6);
    assert_eq!(v.iter().map(|&x| x).sum(), 55);
    assert_eq!(v.slice(0, 0).iter().map(|&x| x).sum(), 0);
}

#[test]
fn test_iterator_product() {
    let v = &[0i, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    assert_eq!(v.slice(0, 4).iter().map(|&x| x).product(), 0);
    assert_eq!(v.slice(1, 5).iter().map(|&x| x).product(), 24);
    assert_eq!(v.slice(0, 0).iter().map(|&x| x).product(), 1);
}

#[test]
fn test_iterator_max() {
    let v = &[0i, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    assert_eq!(v.slice(0, 4).iter().map(|&x| x).max(), Some(3));
    assert_eq!(v.iter().map(|&x| x).max(), Some(10));
    assert_eq!(v.slice(0, 0).iter().map(|&x| x).max(), None);
}

#[test]
fn test_iterator_min() {
    let v = &[0i, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    assert_eq!(v.slice(0, 4).iter().map(|&x| x).min(), Some(0));
    assert_eq!(v.iter().map(|&x| x).min(), Some(0));
    assert_eq!(v.slice(0, 0).iter().map(|&x| x).min(), None);
}

#[test]
fn test_iterator_size_hint() {
    let c = count(0i, 1);
    let v = &[0i, 1, 2, 3, 4, 5, 6, 7, 8, 9];
    let v2 = &[10i, 11, 12];
    let vi = v.iter();

    assert_eq!(c.size_hint(), (uint::MAX, None));
    assert_eq!(vi.size_hint(), (10, Some(10)));

    assert_eq!(c.take(5).size_hint(), (5, Some(5)));
    assert_eq!(c.skip(5).size_hint().val1(), None);
    assert_eq!(c.take_while(|_| false).size_hint(), (0, None));
    assert_eq!(c.skip_while(|_| false).size_hint(), (0, None));
    assert_eq!(c.enumerate().size_hint(), (uint::MAX, None));
    assert_eq!(c.chain(vi.map(|&i| i)).size_hint(), (uint::MAX, None));
    assert_eq!(c.zip(vi).size_hint(), (10, Some(10)));
    assert_eq!(c.scan(0i, |_,_| Some(0i)).size_hint(), (0, None));
    assert_eq!(c.filter(|_| false).size_hint(), (0, None));
    assert_eq!(c.map(|_| 0i).size_hint(), (uint::MAX, None));
    assert_eq!(c.filter_map(|_| Some(0i)).size_hint(), (0, None));

    assert_eq!(vi.take(5).size_hint(), (5, Some(5)));
    assert_eq!(vi.take(12).size_hint(), (10, Some(10)));
    assert_eq!(vi.skip(3).size_hint(), (7, Some(7)));
    assert_eq!(vi.skip(12).size_hint(), (0, Some(0)));
    assert_eq!(vi.take_while(|_| false).size_hint(), (0, Some(10)));
    assert_eq!(vi.skip_while(|_| false).size_hint(), (0, Some(10)));
    assert_eq!(vi.enumerate().size_hint(), (10, Some(10)));
    assert_eq!(vi.chain(v2.iter()).size_hint(), (13, Some(13)));
    assert_eq!(vi.zip(v2.iter()).size_hint(), (3, Some(3)));
    assert_eq!(vi.scan(0i, |_,_| Some(0i)).size_hint(), (0, Some(10)));
    assert_eq!(vi.filter(|_| false).size_hint(), (0, Some(10)));
    assert_eq!(vi.map(|i| i+1).size_hint(), (10, Some(10)));
    assert_eq!(vi.filter_map(|_| Some(0i)).size_hint(), (0, Some(10)));
}

#[test]
fn test_collect() {
    let a = vec![1i, 2, 3, 4, 5];
    let b: Vec<int> = a.iter().map(|&x| x).collect();
    assert!(a == b);
}

#[test]
fn test_all() {
    let v: Box<[int]> = box [1i, 2, 3, 4, 5];
    assert!(v.iter().all(|&x| x < 10));
    assert!(!v.iter().all(|&x| x % 2 == 0));
    assert!(!v.iter().all(|&x| x > 100));
    assert!(v.slice(0, 0).iter().all(|_| fail!()));
}

#[test]
fn test_any() {
    let v: Box<[int]> = box [1i, 2, 3, 4, 5];
    assert!(v.iter().any(|&x| x < 10));
    assert!(v.iter().any(|&x| x % 2 == 0));
    assert!(!v.iter().any(|&x| x > 100));
    assert!(!v.slice(0, 0).iter().any(|_| fail!()));
}

#[test]
fn test_find() {
    let v: &[int] = &[1i, 3, 9, 27, 103, 14, 11];
    assert_eq!(*v.iter().find(|x| *x & 1 == 0).unwrap(), 14);
    assert_eq!(*v.iter().find(|x| *x % 3 == 0).unwrap(), 3);
    assert!(v.iter().find(|x| *x % 12 == 0).is_none());
}

#[test]
fn test_position() {
    let v = &[1i, 3, 9, 27, 103, 14, 11];
    assert_eq!(v.iter().position(|x| *x & 1 == 0).unwrap(), 5);
    assert_eq!(v.iter().position(|x| *x % 3 == 0).unwrap(), 1);
    assert!(v.iter().position(|x| *x % 12 == 0).is_none());
}

#[test]
fn test_count() {
    let xs = &[1i, 2, 2, 1, 5, 9, 0, 2];
    assert_eq!(xs.iter().filter(|x| **x == 2).count(), 3);
    assert_eq!(xs.iter().filter(|x| **x == 5).count(), 1);
    assert_eq!(xs.iter().filter(|x| **x == 95).count(), 0);
}

#[test]
fn test_max_by() {
    let xs: &[int] = &[-3i, 0, 1, 5, -10];
    assert_eq!(*xs.iter().max_by(|x| x.abs()).unwrap(), -10);
}

#[test]
fn test_min_by() {
    let xs: &[int] = &[-3i, 0, 1, 5, -10];
    assert_eq!(*xs.iter().min_by(|x| x.abs()).unwrap(), 0);
}

#[test]
fn test_by_ref() {
    let mut xs = range(0i, 10);
    // sum the first five values
    let partial_sum = xs.by_ref().take(5).fold(0, |a, b| a + b);
    assert_eq!(partial_sum, 10);
    assert_eq!(xs.next(), Some(5));
}

#[test]
fn test_rev() {
    let xs = [2i, 4, 6, 8, 10, 12, 14, 16];
    let mut it = xs.iter();
    it.next();
    it.next();
    assert!(it.rev().map(|&x| x).collect::<Vec<int>>() ==
            vec![16, 14, 12, 10, 8, 6]);
}

#[test]
fn test_double_ended_map() {
    let xs = [1i, 2, 3, 4, 5, 6];
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
    let xs = [1i, 2, 3, 4, 5, 6];
    let mut it = xs.iter().map(|&x| x).enumerate();
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
    let xs = [1i, 2, 3, 4, 5, 6];
    let ys = [1i, 2, 3, 7];
    let a = xs.iter().map(|&x| x);
    let b = ys.iter().map(|&x| x);
    let mut it = a.zip(b);
    assert_eq!(it.next(), Some((1, 1)));
    assert_eq!(it.next(), Some((2, 2)));
    assert_eq!(it.next_back(), Some((4, 7)));
    assert_eq!(it.next_back(), Some((3, 3)));
    assert_eq!(it.next(), None);
}

#[test]
fn test_double_ended_filter() {
    let xs = [1i, 2, 3, 4, 5, 6];
    let mut it = xs.iter().filter(|&x| *x & 1 == 0);
    assert_eq!(it.next_back().unwrap(), &6);
    assert_eq!(it.next_back().unwrap(), &4);
    assert_eq!(it.next().unwrap(), &2);
    assert_eq!(it.next_back(), None);
}

#[test]
fn test_double_ended_filter_map() {
    let xs = [1i, 2, 3, 4, 5, 6];
    let mut it = xs.iter().filter_map(|&x| if x & 1 == 0 { Some(x * 2) } else { None });
    assert_eq!(it.next_back().unwrap(), 12);
    assert_eq!(it.next_back().unwrap(), 8);
    assert_eq!(it.next().unwrap(), 4);
    assert_eq!(it.next_back(), None);
}

#[test]
fn test_double_ended_chain() {
    let xs = [1i, 2, 3, 4, 5];
    let ys = [7i, 9, 11];
    let mut it = xs.iter().chain(ys.iter()).rev();
    assert_eq!(it.next().unwrap(), &11)
    assert_eq!(it.next().unwrap(), &9)
    assert_eq!(it.next_back().unwrap(), &1)
    assert_eq!(it.next_back().unwrap(), &2)
    assert_eq!(it.next_back().unwrap(), &3)
    assert_eq!(it.next_back().unwrap(), &4)
    assert_eq!(it.next_back().unwrap(), &5)
    assert_eq!(it.next_back().unwrap(), &7)
    assert_eq!(it.next_back(), None)
}

#[test]
fn test_rposition() {
    fn f(xy: &(int, char)) -> bool { let (_x, y) = *xy; y == 'b' }
    fn g(xy: &(int, char)) -> bool { let (_x, y) = *xy; y == 'd' }
    let v = [(0i, 'a'), (1, 'b'), (2, 'c'), (3, 'b')];

    assert_eq!(v.iter().rposition(f), Some(3u));
    assert!(v.iter().rposition(g).is_none());
}

#[test]
#[should_fail]
fn test_rposition_fail() {
    use std::gc::GC;
    let v = [(box 0i, box(GC) 0i), (box 0i, box(GC) 0i),
             (box 0i, box(GC) 0i), (box 0i, box(GC) 0i)];
    let mut i = 0i;
    v.iter().rposition(|_elt| {
        if i == 2 {
            fail!()
        }
        i += 1;
        false
    });
}


#[cfg(test)]
fn check_randacc_iter<A: PartialEq, T: Clone + RandomAccessIterator<A>>(a: T, len: uint)
{
    let mut b = a.clone();
    assert_eq!(len, b.indexable());
    let mut n = 0u;
    for (i, elt) in a.enumerate() {
        assert!(Some(elt) == b.idx(i));
        n += 1;
    }
    assert_eq!(n, len);
    assert!(None == b.idx(n));
    // call recursively to check after picking off an element
    if len > 0 {
        b.next();
        check_randacc_iter(b, len-1);
    }
}


#[test]
fn test_double_ended_flat_map() {
    let u = [0u,1];
    let v = [5u,6,7,8];
    let mut it = u.iter().flat_map(|x| v.slice(*x, v.len()).iter());
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
fn test_random_access_chain() {
    let xs = [1i, 2, 3, 4, 5];
    let ys = [7i, 9, 11];
    let mut it = xs.iter().chain(ys.iter());
    assert_eq!(it.idx(0).unwrap(), &1);
    assert_eq!(it.idx(5).unwrap(), &7);
    assert_eq!(it.idx(7).unwrap(), &11);
    assert!(it.idx(8).is_none());

    it.next();
    it.next();
    it.next_back();

    assert_eq!(it.idx(0).unwrap(), &3);
    assert_eq!(it.idx(4).unwrap(), &9);
    assert!(it.idx(6).is_none());

    check_randacc_iter(it, xs.len() + ys.len() - 3);
}

#[test]
fn test_random_access_enumerate() {
    let xs = [1i, 2, 3, 4, 5];
    check_randacc_iter(xs.iter().enumerate(), xs.len());
}

#[test]
fn test_random_access_rev() {
    let xs = [1i, 2, 3, 4, 5];
    check_randacc_iter(xs.iter().rev(), xs.len());
    let mut it = xs.iter().rev();
    it.next();
    it.next_back();
    it.next();
    check_randacc_iter(it, xs.len() - 3);
}

#[test]
fn test_random_access_zip() {
    let xs = [1i, 2, 3, 4, 5];
    let ys = [7i, 9, 11];
    check_randacc_iter(xs.iter().zip(ys.iter()), cmp::min(xs.len(), ys.len()));
}

#[test]
fn test_random_access_take() {
    let xs = [1i, 2, 3, 4, 5];
    let empty: &[int] = [];
    check_randacc_iter(xs.iter().take(3), 3);
    check_randacc_iter(xs.iter().take(20), xs.len());
    check_randacc_iter(xs.iter().take(0), 0);
    check_randacc_iter(empty.iter().take(2), 0);
}

#[test]
fn test_random_access_skip() {
    let xs = [1i, 2, 3, 4, 5];
    let empty: &[int] = [];
    check_randacc_iter(xs.iter().skip(2), xs.len() - 2);
    check_randacc_iter(empty.iter().skip(2), 0);
}

#[test]
fn test_random_access_inspect() {
    let xs = [1i, 2, 3, 4, 5];

    // test .map and .inspect that don't implement Clone
    let mut it = xs.iter().inspect(|_| {});
    assert_eq!(xs.len(), it.indexable());
    for (i, elt) in xs.iter().enumerate() {
        assert_eq!(Some(elt), it.idx(i));
    }

}

#[test]
fn test_random_access_map() {
    let xs = [1i, 2, 3, 4, 5];

    let mut it = xs.iter().map(|x| *x);
    assert_eq!(xs.len(), it.indexable());
    for (i, elt) in xs.iter().enumerate() {
        assert_eq!(Some(*elt), it.idx(i));
    }
}

#[test]
fn test_random_access_cycle() {
    let xs = [1i, 2, 3, 4, 5];
    let empty: &[int] = [];
    check_randacc_iter(xs.iter().cycle().take(27), 27);
    check_randacc_iter(empty.iter().cycle(), 0);
}

#[test]
fn test_double_ended_range() {
    assert!(range(11i, 14).rev().collect::<Vec<int>>() == vec![13i, 12, 11]);
    for _ in range(10i, 0).rev() {
        fail!("unreachable");
    }

    assert!(range(11u, 14).rev().collect::<Vec<uint>>() == vec![13u, 12, 11]);
    for _ in range(10u, 0).rev() {
        fail!("unreachable");
    }
}

#[test]
fn test_range() {
    /// A mock type to check Range when ToPrimitive returns None
    struct Foo;

    impl ToPrimitive for Foo {
        fn to_i64(&self) -> Option<i64> { None }
        fn to_u64(&self) -> Option<u64> { None }
    }

    impl Add<Foo, Foo> for Foo {
        fn add(&self, _: &Foo) -> Foo {
            Foo
        }
    }

    impl PartialEq for Foo {
        fn eq(&self, _: &Foo) -> bool {
            true
        }
    }

    impl PartialOrd for Foo {
        fn partial_cmp(&self, _: &Foo) -> Option<Ordering> {
            None
        }
    }

    impl Clone for Foo {
        fn clone(&self) -> Foo {
            Foo
        }
    }

    impl Mul<Foo, Foo> for Foo {
        fn mul(&self, _: &Foo) -> Foo {
            Foo
        }
    }

    impl num::One for Foo {
        fn one() -> Foo {
            Foo
        }
    }

    assert!(range(0i, 5).collect::<Vec<int>>() == vec![0i, 1, 2, 3, 4]);
    assert!(range(-10i, -1).collect::<Vec<int>>() ==
               vec![-10, -9, -8, -7, -6, -5, -4, -3, -2]);
    assert!(range(0i, 5).rev().collect::<Vec<int>>() == vec![4, 3, 2, 1, 0]);
    assert_eq!(range(200i, -5).count(), 0);
    assert_eq!(range(200i, -5).rev().count(), 0);
    assert_eq!(range(200i, 200).count(), 0);
    assert_eq!(range(200i, 200).rev().count(), 0);

    assert_eq!(range(0i, 100).size_hint(), (100, Some(100)));
    // this test is only meaningful when sizeof uint < sizeof u64
    assert_eq!(range(uint::MAX - 1, uint::MAX).size_hint(), (1, Some(1)));
    assert_eq!(range(-10i, -1).size_hint(), (9, Some(9)));
    assert_eq!(range(Foo, Foo).size_hint(), (0, None));
}

#[test]
fn test_range_inclusive() {
    assert!(range_inclusive(0i, 5).collect::<Vec<int>>() ==
            vec![0i, 1, 2, 3, 4, 5]);
    assert!(range_inclusive(0i, 5).rev().collect::<Vec<int>>() ==
            vec![5i, 4, 3, 2, 1, 0]);
    assert_eq!(range_inclusive(200i, -5).count(), 0);
    assert_eq!(range_inclusive(200i, -5).rev().count(), 0);
    assert!(range_inclusive(200i, 200).collect::<Vec<int>>() == vec![200]);
    assert!(range_inclusive(200i, 200).rev().collect::<Vec<int>>() == vec![200]);
}

#[test]
fn test_range_step() {
    assert!(range_step(0i, 20, 5).collect::<Vec<int>>() ==
            vec![0, 5, 10, 15]);
    assert!(range_step(20i, 0, -5).collect::<Vec<int>>() ==
            vec![20, 15, 10, 5]);
    assert!(range_step(20i, 0, -6).collect::<Vec<int>>() ==
            vec![20, 14, 8, 2]);
    assert!(range_step(200u8, 255, 50).collect::<Vec<u8>>() ==
            vec![200u8, 250]);
    assert!(range_step(200i, -5, 1).collect::<Vec<int>>() == vec![]);
    assert!(range_step(200i, 200, 1).collect::<Vec<int>>() == vec![]);
}

#[test]
fn test_range_step_inclusive() {
    assert!(range_step_inclusive(0i, 20, 5).collect::<Vec<int>>() ==
            vec![0, 5, 10, 15, 20]);
    assert!(range_step_inclusive(20i, 0, -5).collect::<Vec<int>>() ==
            vec![20, 15, 10, 5, 0]);
    assert!(range_step_inclusive(20i, 0, -6).collect::<Vec<int>>() ==
            vec![20, 14, 8, 2]);
    assert!(range_step_inclusive(200u8, 255, 50).collect::<Vec<u8>>() ==
            vec![200u8, 250]);
    assert!(range_step_inclusive(200i, -5, 1).collect::<Vec<int>>() ==
            vec![]);
    assert!(range_step_inclusive(200i, 200, 1).collect::<Vec<int>>() ==
            vec![200]);
}

#[test]
fn test_reverse() {
    let mut ys = [1i, 2, 3, 4, 5];
    ys.iter_mut().reverse_();
    assert!(ys == [5, 4, 3, 2, 1]);
}

#[test]
fn test_peekable_is_empty() {
    let a = [1i];
    let mut it = a.iter().peekable();
    assert!( !it.is_empty() );
    it.next();
    assert!( it.is_empty() );
}

#[test]
fn test_min_max() {
    let v: [int, ..0] = [];
    assert_eq!(v.iter().min_max(), NoElements);

    let v = [1i];
    assert!(v.iter().min_max() == OneElement(&1));

    let v = [1i, 2, 3, 4, 5];
    assert!(v.iter().min_max() == MinMax(&1, &5));

    let v = [1i, 2, 3, 4, 5, 6];
    assert!(v.iter().min_max() == MinMax(&1, &6));

    let v = [1i, 1, 1, 1];
    assert!(v.iter().min_max() == MinMax(&1, &1));
}

#[test]
fn test_min_max_result() {
    let r: MinMaxResult<int> = NoElements;
    assert_eq!(r.into_option(), None)

    let r = OneElement(1i);
    assert_eq!(r.into_option(), Some((1,1)));

    let r = MinMax(1i,2);
    assert_eq!(r.into_option(), Some((1,2)));
}

#[test]
fn test_iterate() {
    let mut it = iterate(1u, |x| x * 2);
    assert_eq!(it.next(), Some(1u));
    assert_eq!(it.next(), Some(2u));
    assert_eq!(it.next(), Some(4u));
    assert_eq!(it.next(), Some(8u));
}

#[bench]
fn bench_rposition(b: &mut Bencher) {
    let it: Vec<uint> = range(0u, 300).collect();
    b.iter(|| {
        it.iter().rposition(|&x| x <= 150);
    });
}

#[bench]
fn bench_skip_while(b: &mut Bencher) {
    b.iter(|| {
        let it = range(0u, 100);
        let mut sum = 0;
        it.skip_while(|&x| { sum += x; sum < 4000 }).all(|_| true);
    });
}

#[bench]
fn bench_multiple_take(b: &mut Bencher) {
    let mut it = range(0u, 42).cycle();
    b.iter(|| {
        let n = it.next().unwrap();
        for m in range(0u, n) {
            it.take(it.next().unwrap()).all(|_| true);
        }
    });
}
