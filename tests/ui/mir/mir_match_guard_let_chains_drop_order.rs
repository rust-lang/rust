//@ run-pass
//@ needs-unwind
//@ revisions: edition2021 edition2024
//@ [edition2021] edition: 2021
//@ [edition2024] edition: 2024

// See `mir_drop_order.rs` for more information

#![feature(if_let_guard)]
#![allow(irrefutable_let_patterns)]

use std::cell::RefCell;
use std::panic;

pub struct DropLogger<'a, T> {
    extra: T,
    id: usize,
    log: &'a panic::AssertUnwindSafe<RefCell<Vec<usize>>>,
}

impl<'a, T> Drop for DropLogger<'a, T> {
    fn drop(&mut self) {
        self.log.0.borrow_mut().push(self.id);
    }
}

struct InjectedFailure;

#[allow(unreachable_code)]
fn main() {
    let log = panic::AssertUnwindSafe(RefCell::new(vec![]));
    let d = |id, extra| DropLogger { extra, id: id, log: &log };
    let get = || -> Vec<_> {
        let mut m = log.0.borrow_mut();
        let n = m.drain(..);
        n.collect()
    };

    {
        let _x = (
            d(
                0,
                d(
                    1,
                    match () { () if let Some(_) = d(2, Some(true)).extra
                        && let DropLogger { .. } = d(3, None) => {
                            None
                        }
                        _ => {
                            Some(true)
                        }
                    }
                )
                .extra,
            ),
            d(4, None),
            &d(5, None),
            d(6, None),
            match () {
                () if let DropLogger { .. } = d(7, None)
                && let DropLogger { .. } = d(8, None) => {
                    d(9, None)
                }
                _ => {
                    // 10 is not constructed
                    d(10, None)
                }
            },
        );
        assert_eq!(get(), vec![3, 2, 8, 7, 1]);
    }
    assert_eq!(get(), vec![0, 4, 6, 9, 5]);

    let _ = std::panic::catch_unwind(|| {
        (
            d(
                11,
                d(
                    12,
                    match () {
                        () if let Some(_) = d(13, Some(true)).extra
                                && let DropLogger { .. } = d(14, None) => {
                            None
                        }
                        _ => {
                            Some(true)
                        }
                    }
                )
                .extra,
            ),
            d(15, None),
            &d(16, None),
            d(17, None),
            match () {
                () if let DropLogger { .. } = d(18, None)
                    && let DropLogger { .. } = d(19, None)
                => {
                    d(20, None)
                }
                _ => {
                    // 10 is not constructed
                    d(21, None)
                }
            },
            panic::panic_any(InjectedFailure),
        );
    });
    assert_eq!(get(), vec![14, 13, 19, 18, 20, 17, 15, 11, 16, 12]);
}
