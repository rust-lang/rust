//@ check-pass
//! Regression test for <https://github.com/rust-lang/rust/issues/59333>.
//! A type alias used only as (part of) the self type of an impl was
//! incorrectly flagged as dead code.

#![deny(dead_code)]

struct Runner;

type RuntimeImpl = Runner;

trait Runtime {
    fn run(&mut self);
}

impl Runtime for &mut RuntimeImpl {
    fn run(&mut self) {}
}

struct Walker;

type WalkerImpl = Walker;

trait Walk {
    fn walk(&self) {}
}

impl Walk for WalkerImpl {}

fn main() {
    let mut runner = Runner;
    (&mut runner).run();
    Walker.walk();
}
