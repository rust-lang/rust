#![feature(reentrant_lock)]
//! This is a regression test for
//! <https://rust-lang.zulipchat.com/#narrow/channel/269128-miri/topic/reentrant.20lock.20failure.20on.20musl>.

use std::cell::Cell;
use std::sync::ReentrantLock;
use std::thread;

static LOCK: ReentrantLock<Cell<i32>> = ReentrantLock::new(Cell::new(0));

fn main() {
    for _ in 0..20 {
        thread::spawn(move || {
            let val = LOCK.lock();
            val.set(val.get() + 1);
            drop(val);
        });
    }
}
