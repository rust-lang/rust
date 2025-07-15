// See drop-order-comparisons.rs

//@ edition: 2024
//@ run-pass

#![feature(if_let_guard)]

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

fn t_guard_if_let_chains_then() {
    let e = Events::new();
    _ = match () {
        () if e.ok(1).is_ok()
            && let true = e.ok(9).is_ok()
            && let Ok(_v) = e.ok(8)
            && let Ok(_) = e.ok(7)
            && let Ok(_) = e.ok(6).as_ref()
            && e.ok(2).is_ok()
            && let Ok(_v) = e.ok(5)
            && let Ok(_) = e.ok(4).as_ref() => {
                e.mark(3);
            }
        _ => {}
    };
    e.assert(9);
}

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

fn t_guard_if_let_chains_then_else() {
    let e = Events::new();
    _ = match () {
       () if e.ok(1).is_ok()
            && let true = e.ok(8).is_ok()
            && let Ok(_v) = e.ok(7)
            && let Ok(_) = e.ok(6)
            && let Ok(_) = e.ok(5).as_ref()
            && e.ok(2).is_ok()
            && let Ok(_v) = e.ok(4)
            && let Ok(_) = e.err(3) => {}
        _ => {
            e.mark(9);
        }
    };
    e.assert(9);
}

fn main() {
    t_if_let_chains_then();
    t_guard_if_let_chains_then();
    t_if_let_chains_then_else();
    t_guard_if_let_chains_then_else();
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
    fn err(&self, m: u64) -> Result<LogDrop<'_>, LogDrop<'_>> {
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
