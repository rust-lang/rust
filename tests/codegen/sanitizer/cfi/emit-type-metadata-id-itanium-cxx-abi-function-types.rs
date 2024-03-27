// Verifies that type metadata identifiers for functions are emitted correctly
// for function types.
//
//@ needs-sanitizer-cfi
//@ compile-flags: -Clto -Cno-prepopulate-passes -Copt-level=0 -Zsanitizer=cfi -Ctarget-feature=-crt-static

#![crate_type="lib"]

trait FnSubtrait: Fn() {}

pub fn foo1(_: fn(i32) -> i32) { }
// CHECK: define{{.*}}4foo1{{.*}}!type ![[TYPE1:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo2(_: fn(i32) -> i32, _: fn(i32) -> i32) { }
// CHECK: define{{.*}}4foo2{{.*}}!type ![[TYPE2:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo3(_: fn(i32) -> i32, _: fn(i32) -> i32, _: fn(i32) -> i32) { }
// CHECK: define{{.*}}4foo3{{.*}}!type ![[TYPE3:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo4(_: &dyn Fn(i32) -> i32) { }
// CHECK: define{{.*}}4foo4{{.*}}!type ![[TYPE1]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo5(_: &dyn Fn(i32) -> i32, _: &dyn Fn(i32) -> i32) { }
// CHECK: define{{.*}}4foo5{{.*}}!type ![[TYPE2]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo6(_: &dyn Fn(i32) -> i32, _: &dyn Fn(i32) -> i32, _: &dyn Fn(i32) -> i32) { }
// CHECK: define{{.*}}4foo6{{.*}}!type ![[TYPE3]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo7(_: &dyn FnMut(i32) -> i32) { }
// CHECK: define{{.*}}4foo7{{.*}}!type ![[TYPE1]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo8(_: &dyn FnMut(i32) -> i32, _: &dyn FnMut(i32) -> i32) { }
// CHECK: define{{.*}}4foo8{{.*}}!type ![[TYPE2]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo9(_: &dyn FnMut(i32) -> i32, _: &dyn FnMut(i32) -> i32, _: &dyn FnMut(i32) -> i32) { }
// CHECK: define{{.*}}4foo9{{.*}}!type ![[TYPE3]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo10(_: &dyn FnOnce(i32) -> i32) { }
// CHECK: define{{.*}}5foo10{{.*}}!type ![[TYPE1]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo11(_: &dyn FnOnce(i32) -> i32, _: &dyn FnOnce(i32) -> i32) { }
// CHECK: define{{.*}}5foo11{{.*}}!type ![[TYPE2]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo12(_: &dyn FnOnce(i32) -> i32, _: &dyn FnOnce(i32) -> i32, _: &dyn FnOnce(i32) -> i32) {}
// CHECK: define{{.*}}5foo12{{.*}}!type ![[TYPE3]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo13(_: &dyn FnSubtrait) { }
// CHECK: define{{.*}}5foo13{{.*}}!type ![[TYPE13:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo14(_: &dyn FnSubtrait, _: &dyn FnSubtrait) { }
// CHECK: define{{.*}}5foo14{{.*}}!type ![[TYPE14:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo15(_: &dyn FnSubtrait, _: &dyn FnSubtrait, _: &dyn FnSubtrait) {}
// CHECK: define{{.*}}5foo15{{.*}}!type ![[TYPE15:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}

// CHECK: ![[TYPE1]] = !{i64 0, !"_ZTSFvPFu3i32S_EE"}
// CHECK: ![[TYPE2]] = !{i64 0, !"_ZTSFvPFu3i32S_ES0_E"}
// CHECK: ![[TYPE3]] = !{i64 0, !"_ZTSFvPFu3i32S_ES0_S0_E"}
// CHECK: ![[TYPE13]] = !{i64 0, !"_ZTSFvPFvvEE"}
// CHECK: ![[TYPE14]] = !{i64 0, !"_ZTSFvPFvvES_E"}
// CHECK: ![[TYPE15]] = !{i64 0, !"_ZTSFvPFvvES_S_E"}
