//@ run-pass
#![allow(non_upper_case_globals)]

use std::sync::atomic::{AtomicUsize, Ordering};

static drop_count: AtomicUsize = AtomicUsize::new(0);

struct Foo {
    dropped: bool
}

impl Drop for Foo {
    fn drop(&mut self) {
        // Test to make sure we haven't dropped already
        assert!(!self.dropped);
        self.dropped = true;
        // And record the fact that we dropped for verification later
        drop_count.fetch_add(1, Ordering::Relaxed);
    }
}

pub fn main() {
    // An `if true { expr }` statement should compile the same as `{ expr }`.
    if true {
        let _a = Foo{ dropped: false };
    }
    // Check that we dropped already (as expected from a `{ expr }`).
    assert_eq!(drop_count.load(Ordering::Relaxed), 1);

    // An `if false {} else { expr }` statement should compile the same as `{ expr }`.
    if false {
        panic!();
    } else {
        let _a = Foo{ dropped: false };
    }
    // Check that we dropped already (as expected from a `{ expr }`).
    assert_eq!(drop_count.load(Ordering::Relaxed), 2);
}
