// The const fn interpreter creates a new AllocId every time it evaluates any const.
// If we do that in Miri, repeatedly evaluating a const causes unbounded memory use
// we need to keep track of the base address for that AllocId, and the allocation is never
// deallocated.
// In Miri we explicitly store previously-assigned AllocIds for each const and ensure
// that we only hand out a finite number of AllocIds per const.
// MIR inlining will put every evaluation of the const we're repeatedly evaluating into the same
// stack frame, breaking this test.
//@compile-flags: -Zinline-mir=no

const EVALS: usize = 256;

use std::collections::HashSet;
fn main() {
    let mut addrs = HashSet::new();
    for _ in 0..EVALS {
        addrs.insert(const_addr());
    }
    // Check that the const allocation has multiple base addresses
    assert!(addrs.len() > 1);
    // But also that we get a limited number of unique base addresses
    assert!(addrs.len() < EVALS);

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
