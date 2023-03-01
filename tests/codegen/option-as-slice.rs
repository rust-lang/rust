// compile-flags: -O
// only-x86_64

#![crate_type = "lib"]
#![feature(option_as_slice)]

extern crate core;

use core::num::NonZeroU64;
use core::option::Option;

// CHECK-LABEL: @u64_opt_as_slice
#[no_mangle]
pub fn u64_opt_as_slice(o: &Option<u64>) -> &[u64] {
    // CHECK: start:
    // CHECK-NOT: select
    // CHECK: ret
    o.as_slice()
}

// CHECK-LABEL: @nonzero_u64_opt_as_slice
#[no_mangle]
pub fn nonzero_u64_opt_as_slice(o: &Option<NonZeroU64>) -> &[NonZeroU64] {
    // CHECK: start:
    // CHECK-NOT: select
    // CHECK: ret
    o.as_slice()
}
