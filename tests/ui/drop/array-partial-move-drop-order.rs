//! Array elements are dropped first to last even when an array pattern may have moved some out.
//! Regression test for <https://github.com/rust-lang/rust/issues/164809>.
//@ run-pass
//@ needs-unwind

use std::cell::RefCell;
use std::panic::{AssertUnwindSafe, catch_unwind};

struct D<'a>(u32, &'a RefCell<Vec<u32>>, bool);

impl Drop for D<'_> {
    fn drop(&mut self) {
        self.1.borrow_mut().push(self.0);
        if self.2 {
            panic!();
        }
    }
}

fn main() {
    let log = RefCell::new(vec![]);
    let d = |n| D(n, &log, false);
    let no = std::hint::black_box(false);

    {
        let a = [d(1), d(2)];
        if no {
            let [_x, _] = a;
        }
    }
    assert_eq!(log.take(), [1, 2]);

    {
        let a = [d(1), d(2), d(3)];
        if no {
            let [_x, _, _] = a;
        }
    }
    assert_eq!(log.take(), [1, 2, 3]);

    {
        let a = [d(1), d(2), d(3)];
        if no {
            let [_, _x, _] = a;
        }
    }
    assert_eq!(log.take(), [1, 2, 3]);

    {
        let a = [d(1), d(2), d(3), d(4), d(5)];
        if no {
            let [_, _x, .., _y] = a;
        }
    }
    assert_eq!(log.take(), [1, 2, 3, 4, 5]);

    {
        let a = [d(1), d(2), d(3), d(4)];
        if no {
            let [_x, _y @ .., _] = a;
        }
    }
    assert_eq!(log.take(), [1, 2, 3, 4]);

    {
        let a = [d(1), d(2), d(3)];
        let [_, _x, _] = a;
    }
    assert_eq!(log.take(), [2, 1, 3]);

    let r = catch_unwind(AssertUnwindSafe(|| {
        let a = [d(1), D(2, &log, true), d(3), d(4)];
        if no {
            let [_, _, _x, _] = a;
        }
    }));
    assert!(r.is_err());
    assert_eq!(log.take(), [1, 2, 3, 4]);
}
