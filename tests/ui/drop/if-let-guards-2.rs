//! This demonstrates a surprising behavior in `if_let_guards`, as
//! implemented, that was found by dianne, and also was questioned in
//! the Reference PR by ehuss.
//!
//! See:
//!
//! - https://github.com/rust-lang/rust/pull/141295
//!
//! Author: TC
//! Author: dianne
//! Date: 2025-06-12

//@ run-pass
//@ edition: 2024

#![allow(irrefutable_let_patterns, unreachable_patterns)]

use core::{cell::RefCell, ops::Drop};

fn main() {
    // The unexpected behavior.
    // assert_drop_order(1..=3, |o| match (o.log(2), o.log(1)) {
    //     (_x, _y)
    //         if let _z = o.log(3)
    //             && true => {}
    //     _ => unreachable!(),
    // });
    assert_drop_order(1..=3, |o| match (o.log(2), o.log(3)) {
        (_x, _y)
            if let _z = o.log(1)
                && false =>
        {
            unreachable!()
        }
        _ => (),
    });
    // With temporaries in guard.
    // assert_drop_order(1..=3, |o| match (o.log(2), o.log(1)) {
    //     (_x, _y)
    //         if let _z = o.log(3).as_true()
    //             && true => {}
    //     _ => unreachable!(),
    // });
    assert_drop_order(1..=3, |o| match (o.log(2), o.log(3)) {
        (_x, _y)
            if let _z = o.log(1).as_true()
                && false =>
        {
            unreachable!()
        }
        _ => (),
    });
    // With temporaries in match scrutinee.
    assert_drop_order(1..=3, |o| {
        match (o.log(3).as_true(), o.log(2).as_true()) {
            (_x, _y)
                if let _z = o.log(1).as_true()
                    && true => {}
            _ => unreachable!(),
        }
    });
    assert_drop_order(1..=3, |o| {
        match (o.log(3).as_true(), o.log(2).as_true()) {
            (_x, _y)
                if let _z = o.log(1).as_true()
                    && false =>
            {
                unreachable!()
            }
            _ => (),
        }
    });
    // Control with inner if let.
    assert_drop_order(1..=3, |o| match (o.log(3), o.log(2)) {
        (_x, _y) => {
            if let _z = o.log(1)
                && true
            {}
        }
        _ => unreachable!(),
    });
    assert_drop_order(1..=3, |o| match (o.log(3), o.log(2)) {
        (_x, _y) => {
            if let _z = o.log(1)
                && false
            {
                unreachable!()
            }
        }
        _ => (),
    });
    // Control of drop before next arm.
    assert_drop_order(1..=4, |o| match (o.log(3), o.log(4)) {
        (_x, _y)
            if let _z = o.log(1)
                && false => {}
        _ if let _z = o.log(2) => {}
        _ => unreachable!(),
    });
    // Control of values in guards.
    assert_drop_order(1..=3, |o| match (o.log(3), o.log(2)) {
        (_x, _y)
            if {
                o.log(1);
                true
            } && true => {}
        _ => unreachable!(),
    });
    assert_drop_order(1..=3, |o| match (o.log(2), o.log(3)) {
        (_x, _y)
            if {
                o.log(1);
                true
            } && false =>
        {
            unreachable!()
        }
        _ => (),
    });
    // Control of temporaries in guards.
    assert_drop_order(1..=3, |o| match (o.log(3), o.log(2)) {
        (_x, _y) if o.log(1).as_true() && true => {}
        _ => unreachable!(),
    });
    assert_drop_order(1..=3, |o| match (o.log(2), o.log(3)) {
        (_x, _y) if o.log(1).as_true() && false => {
            unreachable!()
        }
        _ => (),
    });
    // Control of order on guard success.
    assert_drop_order(1..=2, |o| match (o.log(2), o.log(1)) {
        (_x, _y) if true => {}
        _ => unreachable!(),
    });
    assert_drop_order(1..=2, |o| match (o.log(1), o.log(2)) {
        (_x, _y) if false => {
            unreachable!()
        }
        _ => (),
    });
    // Control of order on guard success when used.
    assert_drop_order(1..=2, |o| match (o.log(2), o.log(1)) {
        (_x, _y)
            if {
                _ = (&_x, &_y);
                true
            } && true => {}
        _ => unreachable!(),
    });
    assert_drop_order(1..=2, |o| match (o.log(1), o.log(2)) {
        (_x, _y)
            if {
                _ = (&_x, &_y);
                true
            } && false =>
        {
            unreachable!()
        }
        _ => (),
    });
    // Control of temporaries in match scrutinee.
    assert_drop_order(1..=2, |o| {
        match (o.log(2).as_true(), o.log(1).as_true()) {
            (_x, _y) if true => {}
            _ => unreachable!(),
        }
    });
    assert_drop_order(1..=2, |o| {
        match (o.log(2).as_true(), o.log(1).as_true()) {
            (_x, _y) if false => {
                unreachable!()
            }
            _ => (),
        }
    });
    // Control of issue 142057.
    //
    // See:
    //
    // - https://github.com/rust-lang/rust/issues/142057
    assert_drop_order(1..=2, |o| {
        match (o.log(1), o.log(2).as_true()) {
            (mut _x, ref _y) if true => {}
            _ => unreachable!(),
        }
    });
    assert_drop_order(1..=2, |o| {
        match (o.log(1), o.log(2).as_true()) {
            (mut _x, ref _y) => {}
            _ => unreachable!(),
        }
    });
}

// # Test scaffolding...

struct DropOrder(RefCell<Vec<u64>>);
struct LogDrop<'o>(&'o DropOrder, u64);

impl DropOrder {
    fn log(&self, n: u64) -> LogDrop<'_> {
        LogDrop(self, n)
    }
}

impl<'o> LogDrop<'o> {
    fn as_true(&self) -> bool {
        true
    }
}

impl<'o> Drop for LogDrop<'o> {
    fn drop(&mut self) {
        self.0 .0.borrow_mut().push(self.1);
    }
}

#[track_caller]
fn assert_drop_order(
    ex: impl IntoIterator<Item = u64>,
    f: impl Fn(&DropOrder),
) {
    let order = DropOrder(RefCell::new(Vec::new()));
    f(&order);
    let order = order.0.into_inner();
    let expected: Vec<u64> = ex.into_iter().collect();
    assert_eq!(order, expected);
}
