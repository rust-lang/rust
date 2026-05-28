//@ test-mir-pass: SimplifyCfg-pre-optimizations
// EMIT_MIR_FOR_EACH_PANIC_STRATEGY
// Retagging (from Stacked Borrows) relies on the array index being a fresh
// temporary, so that side-effects cannot change it.
// Test that this is indeed the case.

unsafe fn foo(z: *mut usize) -> u32 {
    *z = 2;
    99
}

// EMIT_MIR array_index_is_temporary.main.SimplifyCfg-pre-optimizations.after.mir
fn main() {
    // CHECK-LABEL: fn main(
    // CHECK: debug x => [[x:_.*]];
    // CHECK: debug y => [[y:_.*]];
    // CHECK: [[y]] = const 1_usize;
    // CHECK: [[tmp:_.*]] = copy [[y]];
    // CHECK: [[x]][[[tmp]]] =
    let mut x = [42, 43, 44];
    let mut y = 1;
    let z: *mut usize = &mut y;
    x[y] = unsafe { foo(z) };
}
