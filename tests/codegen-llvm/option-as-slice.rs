//@ compile-flags: -Copt-level=3 -Z randomize-layout=no
//@ only-x86_64
#![crate_type = "lib"]

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
    // CHECK: %[[LEN:.+]] = load i64
    // CHECK-SAME: !range ![[META_U64:[0-9]+]],
    // CHECK-SAME: !noundef
    // CHECK-NOT: select
    // CHECK-NOT: br
    // CHECK-NOT: switch
    // CHECK-NOT: icmp
    // CHECK: %[[T0:.+]] = insertvalue { ptr, i64 } poison, ptr %{{.+}}, 0
    // CHECK-NEXT: %[[T1:.+]] = insertvalue { ptr, i64 } %[[T0]], i64 %[[LEN]], 1
    // CHECK-NEXT: ret { ptr, i64 } %[[T1]]
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
    // CHECK-NEXT: %[[LEN:.+]] = zext i1 %[[NZ]] to i64
    // CHECK-NOT: select
    // CHECK-NOT: br
    // CHECK-NOT: switch
    // CHECK-NOT: icmp
    // CHECK: %[[T0:.+]] = insertvalue { ptr, i64 } poison, ptr %o, 0
    // CHECK-NEXT: %[[T1:.+]] = insertvalue { ptr, i64 } %[[T0]], i64 %[[LEN]], 1
    // CHECK-NEXT: ret { ptr, i64 } %[[T1]]
    o.as_slice()
}

// CHECK-LABEL: @u8_opt_as_slice
#[no_mangle]
pub fn u8_opt_as_slice(o: &Option<u8>) -> &[u8] {
    // CHECK-NOT: select
    // CHECK-NOT: br
    // CHECK-NOT: switch
    // CHECK-NOT: icmp
    // CHECK: %[[TAG:.+]] = load i8
    // CHECK-SAME: !range ![[META_U8:[0-9]+]],
    // CHECK-SAME: !noundef
    // CHECK: %[[LEN:.+]] = zext{{.*}} i8 %[[TAG]] to i64
    // CHECK-NOT: select
    // CHECK-NOT: br
    // CHECK-NOT: switch
    // CHECK-NOT: icmp
    // CHECK: %[[T0:.+]] = insertvalue { ptr, i64 } poison, ptr %{{.+}}, 0
    // CHECK-NEXT: %[[T1:.+]] = insertvalue { ptr, i64 } %[[T0]], i64 %[[LEN]], 1
    // CHECK-NEXT: ret { ptr, i64 } %[[T1]]
    o.as_slice()
}

// CHECK: ![[META_U64]] = !{i64 0, i64 2}
// CHECK: ![[META_U8]] = !{i8 0, i8 2}
