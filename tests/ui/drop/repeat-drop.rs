//@ run-pass
//@ needs-unwind

#![allow(dropping_references, dropping_copy_types)]

use std::sync::atomic::{AtomicUsize, Ordering};

static CHECK: AtomicUsize = AtomicUsize::new(0);

struct DropChecker(usize);

impl Drop for DropChecker {
    fn drop(&mut self) {
        if CHECK.load(Ordering::Relaxed) != self.0 - 1 {
            panic!("Found {}, should have found {}", CHECK.load(Ordering::Relaxed), self.0 - 1);
        }
        CHECK.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |_c| {
            Some(self.0)
        }).unwrap();
    }
}

macro_rules! check_drops {
    ($l:literal) => {
        assert_eq!(CHECK.load(Ordering::Relaxed), $l)
    };
}

struct DropPanic;

impl Drop for DropPanic {
    fn drop(&mut self) {
        panic!()
    }
}

fn value_zero() {
    CHECK.store(0, Ordering::Relaxed);
    let foo = DropChecker(1);
    let v: [DropChecker; 0] = [foo; 0];
    check_drops!(1);
    std::mem::drop(v);
    check_drops!(1);
}

fn value_one() {
    CHECK.store(0, Ordering::Relaxed);
    let foo = DropChecker(1);
    let v: [DropChecker; 1] = [foo; 1];
    check_drops!(0);
    std::mem::drop(v);
    check_drops!(1);
}

const DROP_CHECKER: DropChecker = DropChecker(1);

fn const_zero() {
    CHECK.store(0, Ordering::Relaxed);
    let v: [DropChecker; 0] = [DROP_CHECKER; 0];
    check_drops!(0);
    std::mem::drop(v);
    check_drops!(0);
}

fn const_one() {
    CHECK.store(0, Ordering::Relaxed);
    let v: [DropChecker; 1] = [DROP_CHECKER; 1];
    check_drops!(0);
    std::mem::drop(v);
    check_drops!(1);
}

fn const_generic_zero<const N: usize>() {
    CHECK.store(0, Ordering::Relaxed);
    let v: [DropChecker; N] = [DROP_CHECKER; N];
    check_drops!(0);
    std::mem::drop(v);
    check_drops!(0);
}

fn const_generic_one<const N: usize>() {
    CHECK.store(0, Ordering::Relaxed);
    let v: [DropChecker; N] = [DROP_CHECKER; N];
    check_drops!(0);
    std::mem::drop(v);
    check_drops!(1);
}

// Make sure that things are allowed to promote as expected

fn allow_promote() {
    CHECK.store(0, Ordering::Relaxed);
    let foo = DropChecker(1);
    let v: &'static [DropChecker; 0] = &[foo; 0];
    check_drops!(1);
    std::mem::drop(v);
    check_drops!(1);
}

// Verify that unwinding in the drop causes the right things to drop in the right order
fn on_unwind() {
    CHECK.store(0, Ordering::Relaxed);
    std::panic::catch_unwind(|| {
        let panic = DropPanic;
        let _local = DropChecker(2);
        let _v = (DropChecker(1), [panic; 0]);
        std::process::abort();
    })
    .unwrap_err();
    check_drops!(2);
}

fn main() {
    value_zero();
    value_one();
    const_zero();
    const_one();
    const_generic_zero::<0>();
    const_generic_one::<1>();
    allow_promote();
    on_unwind();
}
