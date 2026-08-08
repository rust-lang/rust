// Verifies that thread locals can be accessed (i.e., through the
// compiler-generated accessors for them).
//
//@ needs-sanitizer-kcfi
//@ only-linux
//@ ignore-backends: gcc
//@ compile-flags: -Ctarget-feature=-crt-static -Cpanic=abort -Cprefer-dynamic=off -Copt-level=0 -Zsanitizer=kcfi -Cunsafe-allow-abi-mismatch=sanitizer
//@ run-pass

use std::cell::Cell;

// The ThreadLocalShim for the accessor below is not transformed, as it does not implement any
// trait method and can not be called through a vtable.
thread_local! {
    static COUNTER: Cell<u32> = const { Cell::new(0) };
}

fn main() {
    COUNTER.with(|counter| counter.set(counter.get() + 1));
    assert_eq!(COUNTER.with(|counter| counter.get()), 1);
}
