//@ test-mir-pass: ReferencePropagation
//! Regression test for #132898.
//!
//! `ReferencePropagation` must not collapse a reborrow of a mutable reference
//! (`_2 = &raw mut (*_3)` where `_3 = &mut _1.0`) onto the direct place
//! (`_2 = &raw mut _1.0`) when the root local `_1` has another independent
//! borrow. Doing so shortens the pointer's provenance and introduces UB. The
//! reborrow through `_3` must be preserved.

#![crate_type = "lib"]

struct Foo(u64);

impl Foo {
    fn add(&mut self, n: u64) -> u64 {
        self.0 + n
    }
}

// EMIT_MIR reference_prop_mutable_alias.two_phase.ReferencePropagation.diff
pub fn two_phase() {
    // CHECK-LABEL: fn two_phase(
    // The `&mut f.0` reference must survive: it carries the provenance the write
    // travels through. The pass may still propagate the *raw* reborrow into the
    // direct write (`(*_3) = ...`), which is the pass's intended sound transform;
    // but it must NOT collapse the chain onto the direct place `_1.0 = ...`,
    // which is what introduces aliasing UB w.r.t. the 2-phase `&mut _1` below.
    // CHECK: _{{[0-9]+}} = &mut (_{{[0-9]+}}.0: u64)
    // CHECK: (*_{{[0-9]+}}) = const 42_u64
    // CHECK-NOT: (_{{[0-9]+}}.0: u64) = const 42_u64
    let mut f = Foo(0);
    let alias = &mut f.0 as *mut u64;
    let _ = f.add(unsafe {
        *alias = 42;
        0
    });
}
