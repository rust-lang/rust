//@ run-pass
//@ needs-unwind
//@ ignore-backends: gcc

use std::cell::RefCell;
use std::panic;

pub struct DropLogger<'a> {
    id: usize,
    log: &'a panic::AssertUnwindSafe<RefCell<Vec<usize>>>
}

impl<'a> Drop for DropLogger<'a> {
    fn drop(&mut self) {
        self.log.0.borrow_mut().push(self.id);
    }
}

struct InjectedFailure;

#[allow(unreachable_code)]
fn main() {
    let log = panic::AssertUnwindSafe(RefCell::new(vec![]));
    let d = |id| DropLogger { id: id, log: &log };
    let get = || -> Vec<_> {
        let mut m = log.0.borrow_mut();
        let n = m.drain(..);
        n.collect()
    };

    {
        let _x = (d(0), &d(1), d(2), &d(3));
        // all borrows are extended - nothing has been dropped yet
        assert_eq!(get(), vec![]);
    }
    // in a let-statement, extended places are dropped
    // *after* the let result (tho they have the same scope
    // as far as scope-based borrowck goes).
    assert_eq!(get(), vec![0, 2, 3, 1]);

    let _ = std::panic::catch_unwind(|| {
        (d(4), &d(5), d(6), &d(7), panic::panic_any(InjectedFailure));
    });

    // here, the temporaries (5/7) live until the end of the
    // containing statement, which is destroyed after the operands
    // (4/6) on a panic.
    assert_eq!(get(), vec![6, 4, 7, 5]);
}
