//@ run-pass

//! Test that `.use` only clones place expressions.

#![feature(ergonomic_clones)]
#![expect(incomplete_features)]

use std::clone::UseCloned;
use std::sync::atomic::{AtomicUsize, Ordering};

static NUM_USES: AtomicUsize = AtomicUsize::new(0);

fn check_num_uses(value: usize) {
    assert_eq!(NUM_USES.swap(0, Ordering::Relaxed), value);
}

struct NoisyClone;

impl Clone for NoisyClone {
    fn clone(&self) -> Self {
        NUM_USES.fetch_add(1, Ordering::Relaxed);
        NoisyClone
    }
}

struct ImplsDrop<T>(T);

impl<T> Drop for ImplsDrop<T> {
    fn drop(&mut self) {}
}

impl UseCloned for NoisyClone {}

fn main() {
    let a = NoisyClone;
    let _b = a.use;
    check_num_uses(1);

    let a = NoisyClone;
    let _b = a.use.use;
    check_num_uses(1);

    let a = NoisyClone;
    let _b = a.use.use;
    check_num_uses(1);

    let _a = NoisyClone.use;
    check_num_uses(0);

    const CONST: NoisyClone = NoisyClone;
    let _a = CONST.use;
    check_num_uses(0);

    let _a = (const { NoisyClone }).use;
    check_num_uses(0);

    static STATIC: NoisyClone = NoisyClone;
    let _a = STATIC.use;
    check_num_uses(1);

    let _a = (|| NoisyClone)().use;
    check_num_uses(0);

    let a = &(NoisyClone,);
    let _b = a.0.use;
    check_num_uses(1);

    let _a = (&(NoisyClone,)).0.use;
    check_num_uses(1);

    let _a = (const { &(NoisyClone,) }).0.use;
    check_num_uses(1);

    let a = NoisyClone;
    let _b = { a }.use;
    check_num_uses(0);

    let a = (NoisyClone,);
    let _b = a.0.use;
    check_num_uses(1);

    const COMPOUND_CONST: (NoisyClone,) = (NoisyClone,);
    let _a = COMPOUND_CONST.0.use;
    check_num_uses(0);

    const COMPOUND_REF_CONST: &(NoisyClone,) = &(NoisyClone,);
    let _a = COMPOUND_REF_CONST.0.use;
    check_num_uses(1);

    static COMPOUND_STATIC: (NoisyClone,) = (NoisyClone,);
    let _a = COMPOUND_STATIC.0.use;
    check_num_uses(1);

    let _a = ImplsDrop(NoisyClone).0.use;
    check_num_uses(1);

    let _a = ImplsDrop((NoisyClone,)).0.0.use;
    check_num_uses(1);

    let _a = (*Box::new(NoisyClone)).use;
    check_num_uses(0);

    let a = Box::new(NoisyClone);
    let _b = (*a).use;
    check_num_uses(1);
}
