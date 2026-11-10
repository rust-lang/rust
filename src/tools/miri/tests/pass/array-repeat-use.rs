//! Test `.use` repeats inside array repeat expressions

#![feature(ergonomic_clones)]
#![expect(incomplete_features)]

use std::clone::UseCloned;
use std::sync::atomic::{AtomicUsize, Ordering};

static NUM_USES: AtomicUsize = AtomicUsize::new(0);
static NUM_DROPS: AtomicUsize = AtomicUsize::new(0);

#[track_caller]
fn check_num_uses(value: usize) {
    assert_eq!(NUM_USES.swap(0, Ordering::Relaxed), value);
}

#[track_caller]
fn check_num_drops(value: usize) {
    assert_eq!(NUM_DROPS.swap(0, Ordering::Relaxed), value);
}

struct NoisyClone(u8);

impl Clone for NoisyClone {
    fn clone(&self) -> Self {
        NUM_USES.fetch_add(1, Ordering::Relaxed);
        NoisyClone(self.0)
    }
}

impl Drop for NoisyClone {
    fn drop(&mut self) {
        NUM_DROPS.fetch_add(1, Ordering::Relaxed);
    }
}

impl UseCloned for NoisyClone {}

fn generic_place<const N: usize>() -> [NoisyClone; N] {
    let a = NoisyClone(42);
    let ret = [a.use; N];
    drop(a);
    check_num_drops(1);
    ret
}

fn generic_value<const N: usize>() -> [NoisyClone; N] {
    [NoisyClone(42).use; N]
}

fn main() {
    let a = NoisyClone(42);

    let b = [a.use; 0];
    check_num_uses(0);
    check_num_drops(0);
    drop(b);
    check_num_drops(0);

    let b = generic_place::<0>();
    check_num_uses(0);
    drop(b);
    check_num_drops(0);

    let b = [a.use; 1];
    check_num_uses(1);
    check_num_drops(0);
    drop(b);
    check_num_drops(1);

    let b = generic_place::<1>();
    check_num_uses(1);
    drop(b);
    check_num_drops(1);

    let b = [a.use; 17];
    check_num_uses(17);
    check_num_drops(0);
    drop(b);
    check_num_drops(17);

    let b = generic_place::<17>();
    check_num_uses(17);
    drop(b);
    check_num_drops(17);

    drop(a);
    check_num_drops(1);

    let b = [NoisyClone(42).use; 0];
    check_num_uses(0);
    check_num_drops(1);
    drop(b);
    check_num_drops(0);

    let b = generic_value::<0>();
    check_num_uses(0);
    check_num_drops(1);
    drop(b);
    check_num_drops(0);

    let b = [NoisyClone(42).use; 1];
    check_num_uses(0);
    check_num_drops(0);
    drop(b);
    check_num_drops(1);

    let b = generic_value::<1>();
    check_num_uses(0);
    check_num_drops(0);
    drop(b);
    check_num_drops(1);

    let b = [NoisyClone(42).use; 17];
    check_num_uses(16);
    check_num_drops(0);
    drop(b);
    check_num_drops(17);

    let b = generic_value::<17>();
    check_num_uses(16);
    check_num_drops(0);
    drop(b);
    check_num_drops(17);
}
