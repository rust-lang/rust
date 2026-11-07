// Verifies that type metadata identifiers for functions are emitted correctly
// for pointer types.
//
//@ needs-sanitizer-cfi
//@ compile-flags: -Clto -Cno-prepopulate-passes -Copt-level=0 -Zsanitizer=cfi -Ctarget-feature=-crt-static -C unsafe-allow-abi-mismatch=sanitizer

#![crate_type = "lib"]

pub fn foo1(_: &mut i32) {}
// CHECK: define{{.*}}4foo1{{.*}}!type ![[TYPE1:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo2(_: &mut i32, _: &i32) {}
// CHECK: define{{.*}}4foo2{{.*}}!type ![[TYPE2:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo3(_: &mut i32, _: &i32, _: &i32) {}
// CHECK: define{{.*}}4foo3{{.*}}!type ![[TYPE3:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo4(_: &i32) {}
// CHECK: define{{.*}}4foo4{{.*}}!type ![[TYPE4:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo5(_: &i32, _: &mut i32) {}
// CHECK: define{{.*}}4foo5{{.*}}!type ![[TYPE5:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo6(_: &i32, _: &mut i32, _: &mut i32) {}
// CHECK: define{{.*}}4foo6{{.*}}!type ![[TYPE6:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo7(_: *mut i32) {}
// CHECK: define{{.*}}4foo7{{.*}}!type ![[TYPE7:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo8(_: *mut i32, _: *const i32) {}
// CHECK: define{{.*}}4foo8{{.*}}!type ![[TYPE8:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo9(_: *mut i32, _: *const i32, _: *const i32) {}
// CHECK: define{{.*}}4foo9{{.*}}!type ![[TYPE9:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo10(_: *const i32) {}
// CHECK: define{{.*}}5foo10{{.*}}!type ![[TYPE10:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo11(_: *const i32, _: *mut i32) {}
// CHECK: define{{.*}}5foo11{{.*}}!type ![[TYPE11:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo12(_: *const i32, _: *mut i32, _: *mut i32) {}
// CHECK: define{{.*}}5foo12{{.*}}!type ![[TYPE12:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo13(_: fn(i32) -> i32) {}
// CHECK: define{{.*}}5foo13{{.*}}!type ![[TYPE13:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo14(_: fn(i32) -> i32, _: fn(i32) -> i32) {}
// CHECK: define{{.*}}5foo14{{.*}}!type ![[TYPE14:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo15(_: fn(i32) -> i32, _: fn(i32) -> i32, _: fn(i32) -> i32) {}
// CHECK: define{{.*}}5foo15{{.*}}!type ![[TYPE15:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}

// CHECK: ![[TYPE1]] = !{i64 0, !"_ZTSFvU3mutu3refIu3i32EE"}
// CHECK: ![[TYPE2]] = !{i64 0, !"_ZTSFvU3mutu3refIu3i32ES0_E"}
// CHECK: ![[TYPE3]] = !{i64 0, !"_ZTSFvU3mutu3refIu3i32ES0_S0_E"}
// CHECK: ![[TYPE4]] = !{i64 0, !"_ZTSFvu3refIu3i32EE"}
// CHECK: ![[TYPE5]] = !{i64 0, !"_ZTSFvu3refIu3i32EU3mutS0_E"}
// CHECK: ![[TYPE6]] = !{i64 0, !"_ZTSFvu3refIu3i32EU3mutS0_S1_E"}
// CHECK: ![[TYPE7]] = !{i64 0, !"_ZTSFvPu3i32E"}
// CHECK: ![[TYPE8]] = !{i64 0, !"_ZTSFvPu3i32PKS_E"}
// CHECK: ![[TYPE9]] = !{i64 0, !"_ZTSFvPu3i32PKS_S2_E"}
// CHECK: ![[TYPE10]] = !{i64 0, !"_ZTSFvPKu3i32E"}
// CHECK: ![[TYPE11]] = !{i64 0, !"_ZTSFvPKu3i32PS_E"}
// CHECK: ![[TYPE12]] = !{i64 0, !"_ZTSFvPKu3i32PS_S2_E"}
// CHECK: ![[TYPE13]] = !{i64 0, !"_ZTSFvPFu3i32S_EE"}
// CHECK: ![[TYPE14]] = !{i64 0, !"_ZTSFvPFu3i32S_ES0_E"}
// CHECK: ![[TYPE15]] = !{i64 0, !"_ZTSFvPFu3i32S_ES0_S0_E"}
