// Verifies that type metadata identifiers for functions are emitted correctly
// for sequence types.
//
//@ needs-sanitizer-cfi
//@ compile-flags: -Clto -Cno-prepopulate-passes -Copt-level=0 -Zsanitizer=cfi -Ctarget-feature=-crt-static -C unsafe-allow-abi-mismatch=sanitizer

#![crate_type = "lib"]

pub fn foo1(_: (i32, i32)) {}
// CHECK: define{{.*}}4foo1{{.*}}!type ![[TYPE1:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo2(_: (i32, i32), _: (i32, i32)) {}
// CHECK: define{{.*}}4foo2{{.*}}!type ![[TYPE2:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo3(_: (i32, i32), _: (i32, i32), _: (i32, i32)) {}
// CHECK: define{{.*}}4foo3{{.*}}!type ![[TYPE3:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo4(_: [i32; 32]) {}
// CHECK: define{{.*}}4foo4{{.*}}!type ![[TYPE4:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo5(_: [i32; 32], _: [i32; 32]) {}
// CHECK: define{{.*}}4foo5{{.*}}!type ![[TYPE5:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo6(_: [i32; 32], _: [i32; 32], _: [i32; 32]) {}
// CHECK: define{{.*}}4foo6{{.*}}!type ![[TYPE6:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo7(_: &[i32]) {}
// CHECK: define{{.*}}4foo7{{.*}}!type ![[TYPE7:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo8(_: &[i32], _: &[i32]) {}
// CHECK: define{{.*}}4foo8{{.*}}!type ![[TYPE8:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo9(_: &[i32], _: &[i32], _: &[i32]) {}
// CHECK: define{{.*}}4foo9{{.*}}!type ![[TYPE9:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}

// CHECK: ![[TYPE1]] = !{i64 0, !"_ZTSFvu5tupleIu3i32S_EE"}
// CHECK: ![[TYPE2]] = !{i64 0, !"_ZTSFvu5tupleIu3i32S_ES0_E"}
// CHECK: ![[TYPE3]] = !{i64 0, !"_ZTSFvu5tupleIu3i32S_ES0_S0_E"}
// CHECK: ![[TYPE4]] = !{i64 0, !"_ZTSFvA32u3i32E"}
// CHECK: ![[TYPE5]] = !{i64 0, !"_ZTSFvA32u3i32S0_E"}
// CHECK: ![[TYPE6]] = !{i64 0, !"_ZTSFvA32u3i32S0_S0_E"}
// CHECK: ![[TYPE7]] = !{i64 0, !"_ZTSFvu3refIu5sliceIu3i32EEE"}
// CHECK: ![[TYPE8]] = !{i64 0, !"_ZTSFvu3refIu5sliceIu3i32EES1_E"}
// CHECK: ![[TYPE9]] = !{i64 0, !"_ZTSFvu3refIu5sliceIu3i32EES1_S1_E"}
