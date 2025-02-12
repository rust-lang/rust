// This tests various aspects of the drop order with a focus on:
//
// - The lifetime of temporaries with the `if let` construct (and with
// various similar constructs) and how these lifetimes were shortened
// for `if let` in Rust 2024.
//
// - The shortening of the lifetimes of temporaries in tail
// expressions in Rust 2024.
//
// - The behavior of `let` chains and how this behavior compares to
// nested `if let` expressions and chained `let .. else` statements.
//
// In the tests below, `Events` tracks a sequence of numbered events.
// Calling `e.mark(..)` logs a numbered event immediately.  Calling
// `e.ok(..)` or `e.err(..)` returns an `Ok(_)` or `Err(_)` value,
// respectively, and logs the numbered event when that value is
// dropped.  Calling `e.assert()` verifies that the correct number of
// events were logged and that they were logged in the correct order.

//@ revisions: e2021 e2024
//@ [e2021] edition: 2021
//@ [e2021] run-rustfix
//@ [e2021] rustfix-only-machine-applicable
//@ [e2024] edition: 2024
//@ run-pass

#![feature(let_chains)]
#![cfg_attr(e2021, warn(rust_2024_compatibility))]

fn t_bindings() {
    let e = Events::new();
    _ = {
        e.mark(1);
        let _v = e.ok(8);
        let _v = e.ok(2).is_ok();
        let _ = e.ok(3);
        let Ok(_) = e.ok(4) else { unreachable!() };
        let Ok(_) = e.ok(5).as_ref() else { unreachable!() };
        let _v = e.ok(7);
        e.mark(6);
    };
    e.assert(8);
}

fn t_tuples() {
    let e = Events::new();
    _ = (e.ok(1), e.ok(4).is_ok(), e.ok(2), e.ok(3).is_ok());
    e.assert(4);
}

fn t_arrays() {
    let e = Events::new();
    trait Tr {}
    impl<T> Tr for T {}
    fn b<'a, T: 'a>(x: T) -> Box<dyn Tr + 'a> {
        Box::new(x)
    }
    _ = [b(e.ok(1)), b(e.ok(4).is_ok()), b(e.ok(2)), b(e.ok(3).is_ok())];
    e.assert(4);
}

fn t_fncalls() {
    let e = Events::new();
    let f = |_, _, _, _| {};
    _ = f(e.ok(2), e.ok(4).is_ok(), e.ok(1), e.ok(3).is_ok());
    e.assert(4);
}

#[cfg(e2021)]
#[rustfmt::skip]
fn t_tailexpr_bindings() {
    let e = Events::new();
    _ = ({
        let _v = e.ok(2);
        let _v = e.ok(1);
        e.ok(5).is_ok()
        //[e2021]~^ WARN relative drop order changing in Rust 2024
        //[e2021]~| WARN this changes meaning in Rust 2024
    }, e.mark(3), e.ok(4));
    e.assert(5);
}

#[cfg(e2024)]
#[rustfmt::skip]
fn t_tailexpr_bindings() {
    let e = Events::new();
    _ = ({
        let _v = e.ok(3);
        let _v = e.ok(2);
        e.ok(1).is_ok()
    }, e.mark(4), e.ok(5));
    e.assert(5);
}

#[cfg(e2021)]
#[rustfmt::skip]
fn t_tailexpr_tuples() {
    let e = Events::new();
    _ = ({
        (e.ok(2), e.ok(6).is_ok(), e.ok(3), e.ok(5).is_ok())
        //[e2021]~^ WARN relative drop order changing in Rust 2024
        //[e2021]~| WARN this changes meaning in Rust 2024
        //[e2021]~| WARN relative drop order changing in Rust 2024
        //[e2021]~| WARN this changes meaning in Rust 2024
    }, e.mark(1), e.ok(4));
    e.assert(6);
}

#[cfg(e2024)]
#[rustfmt::skip]
fn t_tailexpr_tuples() {
    let e = Events::new();
    _ = ({
        (e.ok(4), e.ok(2).is_ok(), e.ok(5), e.ok(1).is_ok())
    }, e.mark(3), e.ok(6));
    e.assert(6);
}

#[cfg(e2021)]
#[rustfmt::skip]
fn t_if_let_then() {
    let e = Events::new();
    _ = (if let Ok(_) = e.ok(4).as_ref() {
            //[e2021]~^ WARN if let` assigns a shorter lifetime since Edition 2024
            //[e2021]~| WARN this changes meaning in Rust 2024
            e.mark(1);
    }, e.mark(2), e.ok(3));
    e.assert(4);
}

#[cfg(e2024)]
#[rustfmt::skip]
fn t_if_let_then() {
    let e = Events::new();
    _ = (if let Ok(_) = e.ok(2).as_ref() {
            e.mark(1);
    }, e.mark(3), e.ok(4));
    e.assert(4);
}

#[cfg(e2021)]
#[rustfmt::skip]
fn t_if_let_else() {
    let e = Events::new();
    _ = (if let Ok(_) = e.err(4).as_ref() {} else {
            //[e2021]~^ WARN if let` assigns a shorter lifetime since Edition 2024
            //[e2021]~| WARN this changes meaning in Rust 2024
            e.mark(1);
    }, e.mark(2), e.ok(3));
    e.assert(4);
}

#[cfg(e2024)]
#[rustfmt::skip]
fn t_if_let_else() {
    let e = Events::new();
    _ = (if let Ok(_) = e.err(1).as_ref() {} else {
            e.mark(2);
    }, e.mark(3), e.ok(4));
    e.assert(4);
}

#[rustfmt::skip]
fn t_match_then() {
    let e = Events::new();
    _ = (match e.ok(4).as_ref() {
            Ok(_) => e.mark(1),
            _ => unreachable!(),
    }, e.mark(2), e.ok(3));
    e.assert(4);
}

#[rustfmt::skip]
fn t_match_else() {
    let e = Events::new();
    _ = (match e.err(4).as_ref() {
            Ok(_) => unreachable!(),
            _ => e.mark(1),
    }, e.mark(2), e.ok(3));
    e.assert(4);
}

#[rustfmt::skip]
fn t_let_else_then() {
    let e = Events::new();
    _ = ('top: {
         'chain: {
            let Ok(_) = e.ok(1).as_ref() else { break 'chain };
            // The "then" branch:
            e.mark(2);
            break 'top;
        }
        // The "else" branch:
        unreachable!()
    }, e.mark(3), e.ok(4));
    e.assert(4);
}

#[rustfmt::skip]
fn t_let_else_else() {
    let e = Events::new();
    _ = ('top: {
         'chain: {
            let Ok(_) = e.err(1).as_ref() else { break 'chain };
            // The "then" branch:
            unreachable!();
            #[allow(unreachable_code)]
            break 'top;
        }
        // The "else" branch:
        e.mark(2);
    }, e.mark(3), e.ok(4));
    e.assert(4);
}

#[cfg(e2021)]
#[rustfmt::skip]
fn t_if_let_then_tailexpr() {
    let e = Events::new();
    _ = ({
        if let Ok(_) = e.ok(4).as_ref() {
            //[e2021]~^ WARN relative drop order changing in Rust 2024
            //[e2021]~| WARN this changes meaning in Rust 2024
            e.mark(1);
        }
    }, e.mark(2), e.ok(3));
    e.assert(4);
}

#[cfg(e2024)]
#[rustfmt::skip]
fn t_if_let_then_tailexpr() {
    let e = Events::new();
    _ = ({
        if let Ok(_) = e.ok(2).as_ref() {
            e.mark(1);
        }
    }, e.mark(3), e.ok(4));
    e.assert(4);
}

#[cfg(e2021)]
#[rustfmt::skip]
fn t_if_let_else_tailexpr() {
    let e = Events::new();
    _ = ({
        if let Ok(_) = e.err(4).as_ref() {} else {
            //[e2021]~^ WARN relative drop order changing in Rust 2024
            //[e2021]~| WARN this changes meaning in Rust 2024
            //[e2021]~| WARN if let` assigns a shorter lifetime since Edition 2024
            //[e2021]~| WARN this changes meaning in Rust 2024
            e.mark(1);
        }
    }, e.mark(2), e.ok(3));
    e.assert(4);
}

#[cfg(e2024)]
#[rustfmt::skip]
fn t_if_let_else_tailexpr() {
    let e = Events::new();
    _ = ({
        if let Ok(_) = e.err(1).as_ref() {} else {
            e.mark(2);
        }
    }, e.mark(3), e.ok(4));
    e.assert(4);
}

#[rustfmt::skip]
fn t_if_let_nested_then() {
    let e = Events::new();
    _ = {
        // The unusual formatting, here and below, is to make the
        // comparison with `let` chains more direct.
        if e.ok(1).is_ok() {
        if let true = e.ok(9).is_ok() {
        if let Ok(_v) = e.ok(8) {
        if let Ok(_) = e.ok(7) {
        if let Ok(_) = e.ok(6).as_ref() {
        if e.ok(2).is_ok() {
        if let Ok(_v) = e.ok(5) {
        if let Ok(_) = e.ok(4).as_ref() {
            e.mark(3);
        }}}}}}}}
    };
    e.assert(9);
}

#[rustfmt::skip]
fn t_let_else_chained_then() {
    let e = Events::new();
    _ = 'top: {
        'chain: {
            if e.ok(1).is_ok() {} else { break 'chain };
            let true = e.ok(2).is_ok() else { break 'chain };
            let Ok(_v) = e.ok(9) else { break 'chain };
            let Ok(_) = e.ok(3) else { break 'chain };
            let Ok(_) = e.ok(4).as_ref() else { break 'chain };
            if e.ok(5).is_ok() {} else { break 'chain };
            let Ok(_v) = e.ok(8) else { break 'chain };
            let Ok(_) = e.ok(6).as_ref() else { break 'chain };
            // The "then" branch:
            e.mark(7);
            break 'top;
        }
        // The "else" branch:
        unreachable!()
    };
    e.assert(9);
}

#[cfg(e2021)]
#[rustfmt::skip]
fn t_if_let_chains_then() {
    let e = Events::new();
    _ = if e.ok(1).is_ok()
        && let true = e.ok(9).is_ok()
        && let Ok(_v) = e.ok(5)
        && let Ok(_) = e.ok(8)
        && let Ok(_) = e.ok(7).as_ref()
        && e.ok(2).is_ok()
        && let Ok(_v) = e.ok(4)
        && let Ok(_) = e.ok(6).as_ref() {
            e.mark(3);
        };
    e.assert(9);
}

#[cfg(e2024)]
#[rustfmt::skip]
fn t_if_let_chains_then() {
    let e = Events::new();
    _ = if e.ok(1).is_ok()
        && let true = e.ok(9).is_ok()
        && let Ok(_v) = e.ok(8)
        && let Ok(_) = e.ok(7)
        && let Ok(_) = e.ok(6).as_ref()
        && e.ok(2).is_ok()
        && let Ok(_v) = e.ok(5)
        && let Ok(_) = e.ok(4).as_ref() {
            e.mark(3);
        };
    e.assert(9);
}

#[cfg(e2021)]
#[rustfmt::skip]
fn t_if_let_nested_else() {
    let e = Events::new();
    _ = if e.err(1).is_ok() {} else {
        if let true = e.err(9).is_ok() {} else {
        //[e2021]~^ WARN if let` assigns a shorter lifetime since Edition 2024
        //[e2021]~| WARN this changes meaning in Rust 2024
        if let Ok(_v) = e.err(8) {} else {
        //[e2021]~^ WARN if let` assigns a shorter lifetime since Edition 2024
        //[e2021]~| WARN this changes meaning in Rust 2024
        if let Ok(_) = e.err(7) {} else {
        //[e2021]~^ WARN if let` assigns a shorter lifetime since Edition 2024
        //[e2021]~| WARN this changes meaning in Rust 2024
        if let Ok(_) = e.err(6).as_ref() {} else {
        //[e2021]~^ WARN if let` assigns a shorter lifetime since Edition 2024
        //[e2021]~| WARN this changes meaning in Rust 2024
        if e.err(2).is_ok() {} else {
        if let Ok(_v) = e.err(5) {} else {
        //[e2021]~^ WARN if let` assigns a shorter lifetime since Edition 2024
        //[e2021]~| WARN this changes meaning in Rust 2024
        if let Ok(_) = e.err(4) {} else {
            //[e2021]~^ WARN if let` assigns a shorter lifetime since Edition 2024
            //[e2021]~| WARN this changes meaning in Rust 2024
            e.mark(3);
        }}}}}}}};
    e.assert(9);
}

#[cfg(e2024)]
#[rustfmt::skip]
fn t_if_let_nested_else() {
    let e = Events::new();
    _ = if e.err(1).is_ok() {} else {
        if let true = e.err(2).is_ok() {} else {
        if let Ok(_v) = e.err(3) {} else {
        if let Ok(_) = e.err(4) {} else {
        if let Ok(_) = e.err(5).as_ref() {} else {
        if e.err(6).is_ok() {} else {
        if let Ok(_v) = e.err(7) {} else {
        if let Ok(_) = e.err(8) {} else {
            e.mark(9);
        }}}}}}}};
    e.assert(9);
}

#[cfg(e2021)]
#[rustfmt::skip]
fn t_if_let_nested_then_else() {
    let e = Events::new();
    _ = if e.ok(1).is_ok() {
        if let true = e.ok(9).is_ok() {
        if let Ok(_v) = e.ok(8) {
        if let Ok(_) = e.ok(7) {
        if let Ok(_) = e.ok(6).as_ref() {
        if e.ok(2).is_ok() {
        if let Ok(_v) = e.ok(5) {
        if let Ok(_) = e.err(4).as_ref() {} else {
            //[e2021]~^ WARN if let` assigns a shorter lifetime since Edition 2024
            //[e2021]~| WARN this changes meaning in Rust 2024
            e.mark(3);
        }}}}}}}};
    e.assert(9);
}

#[cfg(e2024)]
#[rustfmt::skip]
fn t_if_let_nested_then_else() {
    let e = Events::new();
    _ = if e.ok(1).is_ok() {
        if let true = e.ok(9).is_ok() {
        if let Ok(_v) = e.ok(8) {
        if let Ok(_) = e.ok(7) {
        if let Ok(_) = e.ok(6).as_ref() {
        if e.ok(2).is_ok() {
        if let Ok(_v) = e.ok(5) {
        if let Ok(_) = e.err(3).as_ref() {} else {
            e.mark(4);
        }}}}}}}};
    e.assert(9);
}

#[rustfmt::skip]
fn t_let_else_chained_then_else() {
    let e = Events::new();
    _ = 'top: {
        'chain: {
            if e.ok(1).is_ok() {} else { break 'chain };
            let true = e.ok(2).is_ok() else { break 'chain };
            let Ok(_v) = e.ok(8) else { break 'chain };
            let Ok(_) = e.ok(3) else { break 'chain };
            let Ok(_) = e.ok(4).as_ref() else { break 'chain };
            if e.ok(5).is_ok() {} else { break 'chain };
            let Ok(_v) = e.ok(7) else { break 'chain };
            let Ok(_) = e.err(6).as_ref() else { break 'chain };
            // The "then" branch:
            unreachable!();
            #[allow(unreachable_code)]
            break 'top;
        }
        // The "else" branch:
        e.mark(9);
    };
    e.assert(9);
}

#[cfg(e2021)]
#[rustfmt::skip]
fn t_if_let_chains_then_else() {
    let e = Events::new();
    _ = if e.ok(1).is_ok()
        && let true = e.ok(9).is_ok()
        && let Ok(_v) = e.ok(4)
        && let Ok(_) = e.ok(8)
        && let Ok(_) = e.ok(7).as_ref()
        && e.ok(2).is_ok()
        && let Ok(_v) = e.ok(3)
        && let Ok(_) = e.err(6) {} else {
            e.mark(5);
        };
    e.assert(9);
}

#[cfg(e2024)]
#[rustfmt::skip]
fn t_if_let_chains_then_else() {
    let e = Events::new();
    _ = if e.ok(1).is_ok()
        && let true = e.ok(8).is_ok()
        && let Ok(_v) = e.ok(7)
        && let Ok(_) = e.ok(6)
        && let Ok(_) = e.ok(5).as_ref()
        && e.ok(2).is_ok()
        && let Ok(_v) = e.ok(4)
        && let Ok(_) = e.err(3) {} else {
            e.mark(9);
        };
    e.assert(9);
}

fn main() {
    t_bindings();
    t_tuples();
    t_arrays();
    t_fncalls();
    t_tailexpr_bindings();
    t_tailexpr_tuples();
    t_if_let_then();
    t_if_let_else();
    t_match_then();
    t_match_else();
    t_let_else_then();
    t_let_else_else();
    t_if_let_then_tailexpr();
    t_if_let_else_tailexpr();
    t_if_let_nested_then();
    t_let_else_chained_then();
    t_if_let_chains_then();
    t_if_let_nested_else();
    t_if_let_nested_then_else();
    t_let_else_chained_then_else();
    t_if_let_chains_then_else();
}

// # Test scaffolding

use core::cell::RefCell;
use std::collections::HashSet;

/// A buffer to track the order of events.
///
/// First, numbered events are logged into this buffer.
///
/// Then, `assert` is called to verify that the correct number of
/// events were logged, and that they were logged in the expected
/// order.
struct Events(RefCell<Option<Vec<u64>>>);

impl Events {
    const fn new() -> Self {
        Self(RefCell::new(Some(Vec::new())))
    }
    #[track_caller]
    fn assert(&self, max: u64) {
        let buf = &self.0;
        let v1 = buf.borrow().as_ref().unwrap().clone();
        let mut v2 = buf.borrow().as_ref().unwrap().clone();
        *buf.borrow_mut() = None;
        v2.sort();
        let uniq_len = v2.iter().collect::<HashSet<_>>().len();
        // Check that the sequence is sorted.
        assert_eq!(v1, v2);
        // Check that there are no duplicates.
        assert_eq!(v2.len(), uniq_len);
        // Check that the length is the expected one.
        assert_eq!(max, uniq_len as u64);
        // Check that the last marker is the expected one.
        assert_eq!(v2.last().unwrap(), &max);
    }
    /// Return an `Ok` value that logs its drop.
    fn ok(&self, m: u64) -> Result<LogDrop<'_>, LogDrop<'_>> {
        Ok(LogDrop(self, m))
    }
    /// Return an `Err` value that logs its drop.
    fn err(&self, m: u64) -> Result<LogDrop, LogDrop> {
        Err(LogDrop(self, m))
    }
    /// Log an event.
    fn mark(&self, m: u64) {
        self.0.borrow_mut().as_mut().unwrap().push(m);
    }
}

impl Drop for Events {
    fn drop(&mut self) {
        if self.0.borrow().is_some() {
            panic!("failed to call `Events::assert()`");
        }
    }
}

/// A type that logs its drop events.
struct LogDrop<'b>(&'b Events, u64);

impl<'b> Drop for LogDrop<'b> {
    fn drop(&mut self) {
        self.0.mark(self.1);
    }
}
