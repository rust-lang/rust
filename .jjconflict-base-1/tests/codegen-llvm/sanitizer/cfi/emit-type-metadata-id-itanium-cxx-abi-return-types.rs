// Verifies that type metadata identifiers for functions are emitted correctly
// for return types.
//
//@ needs-sanitizer-cfi
//@ compile-flags: -Clto -Cno-prepopulate-passes -Copt-level=0 -Zsanitizer=cfi -Ctarget-feature=-crt-static -Cunsafe-allow-abi-mismatch=sanitizer

#![crate_type = "lib"]

extern crate core;
use core::ffi::*;

pub unsafe extern "C" fn foo1() {}
// CHECK: define{{.*}}foo1{{.*}}!type ![[TYPE1:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}

// This should probably have the same CFI type as `foo1`, but it doesn't right now.
pub unsafe extern "C" fn foo2() -> ! {
    loop {}
}
// CHECK: define{{.*}}foo2{{.*}}!type ![[TYPE2:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}

pub unsafe extern "C" fn foo3(arg: *mut c_void) -> *mut c_void {
    arg
}
// CHECK: define{{.*}}foo3{{.*}}!type ![[TYPE3:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}

pub unsafe extern "C" fn foo4(arg: *const c_void) -> *const c_void {
    arg
}
// CHECK: define{{.*}}foo4{{.*}}!type ![[TYPE4:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}

// `*mut ()` has the same CFI type (`_ZTSFPvS_E`) as `*mut c_void` (`foo3`), so it reuses `TYPE3`.
pub unsafe extern "C" fn foo5(arg: *mut ()) -> *mut () {
    arg
}
// CHECK: define{{.*}}foo5{{.*}}!type ![[TYPE3:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}

pub unsafe extern "C" fn foo6(arg: *mut u8) -> *mut u8 {
    arg
}
// CHECK: define{{.*}}foo6{{.*}}!type ![[TYPE6:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}

pub unsafe extern "C" fn foo7(arg: *mut i8) -> *mut i8 {
    arg
}
// CHECK: define{{.*}}foo7{{.*}}!type ![[TYPE7:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}

// CHECK: ![[TYPE1]] = !{i64 0, !"_ZTSFvvE"}
// CHECK: ![[TYPE2]] = !{i64 0, !"_ZTSFu5nevervE"}
// CHECK: ![[TYPE3]] = !{i64 0, !"_ZTSFPvS_E"}
// CHECK: ![[TYPE4]] = !{i64 0, !"_ZTSFPKvS0_E"}
// CHECK: ![[TYPE6]] = !{i64 0, !"_ZTSFPu2u8S0_E"}
// CHECK: ![[TYPE7]] = !{i64 0, !"_ZTSFPu2i8S0_E"}
