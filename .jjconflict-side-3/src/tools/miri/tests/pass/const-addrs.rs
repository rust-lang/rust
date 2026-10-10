// The interpreter used to create a new AllocId every time it evaluates any const.
// This caused unbounded memory use in Miri.
// This test verifies that we only create a bounded amount of addresses for any given const.
// In practice, the interpreter always returns the same address, but we *do not guarantee* that.
//@compile-flags: -Zinline-mir=no

const EVALS: usize = 64;

use std::collections::HashSet;
fn main() {
    let mut addrs = HashSet::new();
    for _ in 0..EVALS {
        addrs.insert(const_addr());
    }
    // Check that we always return the same base address for the const allocation.
    assert_eq!(addrs.len(), 1);

    // Check that within a call we always produce the same address
    let mut prev = 0;
    for iter in 0..EVALS {
        let addr = "test".as_bytes().as_ptr().addr();
        if iter > 0 {
            assert_eq!(prev, addr);
        }
        prev = addr;
    }
}

fn const_addr() -> usize {
    "test".as_bytes().as_ptr().addr()
}
