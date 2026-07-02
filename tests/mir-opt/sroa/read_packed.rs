//@ test-mir-pass: ScalarReplacementOfAggregates
//@ compile-flags: -Cpanic=abort
//@ no-prefer-dynamic

#![crate_type = "lib"]
#![feature(core_intrinsics)]

use std::intrinsics::{read_via_copy, transmute_unchecked};

#[repr(packed)]
struct Packed<T>(T);

// EMIT_MIR read_packed.read_unaligned.ScalarReplacementOfAggregates.diff
pub const unsafe fn read_unaligned<T>(ptr: *const T) -> T {
    // CHECK-LABEL: fn read_unaligned(_1: *const T) -> T
    // CHECK: debug packed_ptr => [[PPTR:_.+]];
    // CHECK: debug ((packed_val: Packed<T>).0: T) => [[VAL:_.+]];
    // CHECK: [[TEMP:_.+]] = copy [[PPTR]];
    // CHECK: [[VAL]] = copy ((*{{_.+}}).0: T);
    unsafe {
        let packed_ptr = ptr as *const Packed<T>;
        let packed_val = read_via_copy(packed_ptr);
        // transmute because you can't destructure it in a `const fn`
        transmute_unchecked(packed_val)
    }
}
