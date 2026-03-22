//@ run-pass
//! Test that the Decisive trait enables overloading && and || with short-circuiting.

#![feature(decisive_trait)]

use std::ops::{BitAnd, BitOr, Decisive};

#[derive(Copy, Clone, Debug, PartialEq)]
struct Status(i8);

const DONE: Status = Status(1);
const CONT: Status = Status(0);
const FAIL: Status = Status(-1);

impl Decisive for Status {
    fn is_true(&self) -> bool {
        self.0 != -1
    }
    fn is_false(&self) -> bool {
        self.0 != 1
    }
}

impl BitAnd for Status {
    type Output = Status;
    fn bitand(self, rhs: Status) -> Status {
        rhs
    }
}

impl BitOr for Status {
    type Output = Status;
    fn bitor(self, rhs: Status) -> Status {
        rhs
    }
}

static mut EVAL_COUNT: i32 = 0;

fn done() -> Status {
    unsafe { EVAL_COUNT += 1; }
    DONE
}

fn cont() -> Status {
    unsafe { EVAL_COUNT += 1; }
    CONT
}

fn fail() -> Status {
    unsafe { EVAL_COUNT += 1; }
    FAIL
}

fn reset_count() {
    unsafe { EVAL_COUNT = 0; }
}

fn get_count() -> i32 {
    unsafe { EVAL_COUNT }
}

fn main() {
    // Test &&: short-circuits when LHS is_false (i.e., LHS is not complete)

    // DONE && DONE => evaluates both, returns DONE (from bitand)
    reset_count();
    let r = done() && done();
    assert_eq!(r, DONE);
    assert_eq!(get_count(), 2);

    // DONE && FAIL => evaluates both, returns FAIL (from bitand)
    reset_count();
    let r = done() && fail();
    assert_eq!(r, FAIL);
    assert_eq!(get_count(), 2);

    // FAIL && DONE => short-circuits, returns FAIL (LHS), RHS not evaluated
    reset_count();
    let r = fail() && done();
    assert_eq!(r, FAIL);
    assert_eq!(get_count(), 1); // Only LHS evaluated

    // CONT && DONE => short-circuits, returns CONT (LHS), RHS not evaluated
    reset_count();
    let r = cont() && done();
    assert_eq!(r, CONT);
    assert_eq!(get_count(), 1); // Only LHS evaluated

    // Test ||: short-circuits when LHS is_true (i.e., LHS is not failing)

    // FAIL || DONE => evaluates both, returns DONE (from bitor)
    reset_count();
    let r = fail() || done();
    assert_eq!(r, DONE);
    assert_eq!(get_count(), 2);

    // DONE || FAIL => short-circuits, returns DONE (LHS), RHS not evaluated
    reset_count();
    let r = done() || fail();
    assert_eq!(r, DONE);
    assert_eq!(get_count(), 1); // Only LHS evaluated

    // CONT || DONE => short-circuits, returns CONT (LHS), RHS not evaluated
    reset_count();
    let r = cont() || done();
    assert_eq!(r, CONT);
    assert_eq!(get_count(), 1); // Only LHS evaluated

    // Test chaining: a && b && c
    reset_count();
    let r = done() && done() && fail();
    assert_eq!(r, FAIL);
    assert_eq!(get_count(), 3); // All evaluated

    reset_count();
    let r = fail() && done() && done();
    assert_eq!(r, FAIL);
    assert_eq!(get_count(), 1); // Only first evaluated

    // Test that bool && and || still work
    assert_eq!(true && true, true);
    assert_eq!(true && false, false);
    assert_eq!(false || true, true);
    assert_eq!(false || false, false);

    // Test behavior tree pattern: sequence
    reset_count();
    let r = done() && done() && done();
    assert_eq!(r, DONE);
    assert_eq!(get_count(), 3);

    // Test behavior tree pattern: selector
    reset_count();
    let r = fail() || fail() || done();
    assert_eq!(r, DONE);
    assert_eq!(get_count(), 3);
}
