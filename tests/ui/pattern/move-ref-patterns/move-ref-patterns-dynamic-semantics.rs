//@ run-pass

// This test checks the dynamic semantics and drop order of pattern matching
// where a product pattern has both a by-move and by-ref binding.

use std::cell::RefCell;
use std::rc::Rc;

struct X {
    x: Box<usize>,
    d: DropOrderListPtr,
}

type DropOrderListPtr = Rc<RefCell<Vec<usize>>>;

impl Drop for X {
    fn drop(&mut self) {
        self.d.borrow_mut().push(*self.x);
    }
}

enum DoubleOption<T, U> {
    Some2(T, U),
    _None2,
}

fn main() {
    let d: DropOrderListPtr = <_>::default();
    {
        let mk = |v| X { x: Box::new(v), d: d.clone() };
        let check = |a1: &X, a2, b1: &X, b2| {
            assert_eq!(*a1.x, a2);
            assert_eq!(*b1.x, b2);
        };

        let x = DoubleOption::Some2(mk(1), mk(2));
        match x {
            DoubleOption::Some2(ref a, b) => check(a, 1, &b, 2),
            DoubleOption::_None2 => panic!(),
        }
        let x = DoubleOption::Some2(mk(3), mk(4));
        match x {
            DoubleOption::Some2(a, ref b) => check(&a, 3, b, 4),
            DoubleOption::_None2 => panic!(),
        }
        match DoubleOption::Some2(mk(5), mk(6)) {
            DoubleOption::Some2(ref a, b) => check(a, 5, &b, 6),
            DoubleOption::_None2 => panic!(),
        }
        match DoubleOption::Some2(mk(7), mk(8)) {
            DoubleOption::Some2(a, ref b) => check(&a, 7, b, 8),
            DoubleOption::_None2 => panic!(),
        }
        {
            let (a, ref b) = (mk(9), mk(10));
            let (ref c, d) = (mk(11), mk(12));
            check(&a, 9, b, 10);
            check(c, 11, &d, 12);
        }
        fn fun([a, ref mut b, ref xs @ .., ref c, d]: [X; 6]) {
            assert_eq!(*a.x, 13);
            assert_eq!(*b.x, 14);
            assert_eq!(&[*xs[0].x, *xs[1].x], &[15, 16]);
            assert_eq!(*c.x, 17);
            assert_eq!(*d.x, 18);
        }
        fun([mk(13), mk(14), mk(15), mk(16), mk(17), mk(18)]);

        let lam = |(a, ref b, c, ref mut d): (X, X, X, X)| {
            assert_eq!(*a.x, 19);
            assert_eq!(*b.x, 20);
            assert_eq!(*c.x, 21);
            assert_eq!(*d.x, 22);
        };
        lam((mk(19), mk(20), mk(21), mk(22)));
    }
    let expected = [2, 3, 6, 5, 7, 8, 12, 11, 9, 10, 18, 13, 14, 15, 16, 17, 21, 19, 20, 22, 4, 1];
    assert_eq!(&*d.borrow(), &expected);
}
