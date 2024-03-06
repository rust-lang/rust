//@ run-pass
#![allow(unreachable_code)]
// Issue #21486: Make sure that all structures are dropped, even when
// created via FRU and control-flow breaks in the middle of
// construction.

use std::sync::atomic::{Ordering, AtomicUsize};

#[derive(Debug)]
struct Noisy(u8);
impl Drop for Noisy {
    fn drop(&mut self) {
        // println!("splat #{}", self.0);
        event(self.0);
    }
}

#[allow(dead_code)]
#[derive(Debug)]
struct Foo { n0: Noisy, n1: Noisy }
impl Foo {
    fn vals(&self) -> (u8, u8) { (self.n0.0, self.n1.0) }
}

fn leak_1_ret() -> Foo {
    let _old_foo = Foo { n0: Noisy(1), n1: Noisy(2) };
    Foo { n0: { return Foo { n0: Noisy(3), n1: Noisy(4) } },
          .._old_foo
    };
}

fn leak_2_ret() -> Foo {
    let _old_foo = Foo { n0: Noisy(1), n1: Noisy(2) };
    Foo { n1: { return Foo { n0: Noisy(3), n1: Noisy(4) } },
          .._old_foo
    };
}

// In this case, the control flow break happens *before* we construct
// `Foo(Noisy(1),Noisy(2))`, so there should be no record of it in the
// event log.
fn leak_3_ret() -> Foo {
    let _old_foo = || Foo { n0: Noisy(1), n1: Noisy(2) };
    Foo { n1: { return Foo { n0: Noisy(3), n1: Noisy(4) } },
          .._old_foo()
    };
}

pub fn main() {
    reset_log();
    assert_eq!(leak_1_ret().vals(), (3,4));
    assert_eq!(0x01_02_03_04, event_log());

    reset_log();
    assert_eq!(leak_2_ret().vals(), (3,4));
    assert_eq!(0x01_02_03_04, event_log());

    reset_log();
    assert_eq!(leak_3_ret().vals(), (3,4));
    assert_eq!(0x03_04, event_log());
}

static LOG: AtomicUsize = AtomicUsize::new(0);

fn reset_log() {
    LOG.store(0, Ordering::SeqCst);
}

fn event_log() -> usize {
    LOG.load(Ordering::SeqCst)
}

fn event(tag: u8) {
    let old_log = LOG.load(Ordering::SeqCst);
    let new_log = (old_log << 8) + tag as usize;
    LOG.store(new_log, Ordering::SeqCst);
}
