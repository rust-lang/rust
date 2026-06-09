// Verifies that pointer types are generalized.
//
//@ needs-sanitizer-cfi
//@ compile-flags: -Clto -Cno-prepopulate-passes -Ctarget-feature=-crt-static -Zsanitizer=cfi -Zsanitizer-cfi-generalize-pointers -Copt-level=0 -C unsafe-allow-abi-mismatch=sanitizer

#![crate_type = "lib"]

extern crate core;

pub fn foo0(_: &mut i32) {}
// CHECK: define{{.*}}foo0{{.*}}!type ![[TYPE0:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo1(_: &mut i32, _: &mut i32) {}
// CHECK: define{{.*}}foo1{{.*}}!type ![[TYPE1:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo2(_: &mut i32, _: &mut i32, _: &mut i32) {}
// CHECK: define{{.*}}foo2{{.*}}!type ![[TYPE2:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo3(_: &i32) {}
// CHECK: define{{.*}}foo3{{.*}}!type ![[TYPE3:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo4(_: &i32, _: &i32) {}
// CHECK: define{{.*}}foo4{{.*}}!type ![[TYPE4:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo5(_: &i32, _: &i32, _: &i32) {}
// CHECK: define{{.*}}foo5{{.*}}!type ![[TYPE5:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo6(_: *mut i32) {}
// CHECK: define{{.*}}foo6{{.*}}!type ![[TYPE6:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo7(_: *mut i32, _: *mut i32) {}
// CHECK: define{{.*}}foo7{{.*}}!type ![[TYPE7:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo8(_: *mut i32, _: *mut i32, _: *mut i32) {}
// CHECK: define{{.*}}foo8{{.*}}!type ![[TYPE8:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo9(_: *const i32) {}
// CHECK: define{{.*}}foo9{{.*}}!type ![[TYPE9:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo10(_: *const i32, _: *const i32) {}
// CHECK: define{{.*}}foo10{{.*}}!type ![[TYPE10:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo11(_: *const i32, _: *const i32, _: *const i32) {}
// CHECK: define{{.*}}foo11{{.*}}!type ![[TYPE11:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}}

// CHECK: ![[TYPE0]] = !{i64 0, !"_ZTSFvU3mutu3refIvEE.generalized"}
// CHECK: ![[TYPE1]] = !{i64 0, !"_ZTSFvU3mutu3refIvES0_E.generalized"}
// CHECK: ![[TYPE2]] = !{i64 0, !"_ZTSFvU3mutu3refIvES0_S0_E.generalized"}
// CHECK: ![[TYPE3]] = !{i64 0, !"_ZTSFvu3refIvEE.generalized"}
// CHECK: ![[TYPE4]] = !{i64 0, !"_ZTSFvu3refIvES_E.generalized"}
// CHECK: ![[TYPE5]] = !{i64 0, !"_ZTSFvu3refIvES_S_E.generalized"}
// CHECK: ![[TYPE6]] = !{i64 0, !"_ZTSFvPvE.generalized"}
// CHECK: ![[TYPE7]] = !{i64 0, !"_ZTSFvPvS_E.generalized"}
// CHECK: ![[TYPE8]] = !{i64 0, !"_ZTSFvPvS_S_E.generalized"}
// CHECK: ![[TYPE9]] = !{i64 0, !"_ZTSFvPKvE.generalized"}
// CHECK: ![[TYPE10]] = !{i64 0, !"_ZTSFvPKvS0_E.generalized"}
// CHECK: ![[TYPE11]] = !{i64 0, !"_ZTSFvPKvS0_S0_E.generalized"}
