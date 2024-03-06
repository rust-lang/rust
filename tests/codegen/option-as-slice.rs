//@ compile-flags: -O -Z randomize-layout=no
//@ only-x86_64
//@ ignore-llvm-version: 16.0.0
//  ^-- needs https://reviews.llvm.org/D146149 in 16.0.1
#![crate_type = "lib"]
#![feature(generic_nonzero)]

extern crate core;

use core::num::NonZero;
use core::option::Option;

// CHECK-LABEL: @u64_opt_as_slice
#[no_mangle]
pub fn u64_opt_as_slice(o: &Option<u64>) -> &[u64] {
    // CHECK-NOT: select
    // CHECK-NOT: br
    // CHECK-NOT: switch
    // CHECK-NOT: icmp
    o.as_slice()
}

// CHECK-LABEL: @nonzero_u64_opt_as_slice
#[no_mangle]
pub fn nonzero_u64_opt_as_slice(o: &Option<NonZero<u64>>) -> &[NonZero<u64>] {
    // CHECK-NOT: select
    // CHECK-NOT: br
    // CHECK-NOT: switch
    // CHECK-NOT: icmp
    // CHECK: %[[NZ:.+]] = icmp ne i64 %{{.+}}, 0
    // CHECK-NEXT: zext i1 %[[NZ]] to i64
    // CHECK-NOT: select
    // CHECK-NOT: br
    // CHECK-NOT: switch
    // CHECK-NOT: icmp
    o.as_slice()
}
