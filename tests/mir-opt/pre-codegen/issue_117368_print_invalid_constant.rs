//@needs-deterministic-layouts
// Verify that we do not ICE when printing an invalid constant.
// EMIT_MIR_FOR_EACH_BIT_WIDTH
// EMIT_MIR_FOR_EACH_PANIC_STRATEGY

#![feature(allocator_api)]

use std::alloc::{Allocator, Global, Layout};

// EMIT_MIR issue_117368_print_invalid_constant.main.GVN.diff
fn main() {
    // CHECK-LABEL: fn main(
    // CHECK: debug layout => const Layout
    let layout: Layout = None.unwrap();
    let ptr: *mut u8 = Global.allocate(layout).unwrap().as_ptr() as _;
}
