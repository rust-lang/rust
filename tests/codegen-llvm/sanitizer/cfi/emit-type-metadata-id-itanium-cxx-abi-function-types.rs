// Verifies that type metadata identifiers for functions are emitted correctly
// for function types.
//
//@ needs-sanitizer-cfi
//@ compile-flags: -Clto -Cno-prepopulate-passes -Copt-level=0 -Zsanitizer=cfi -Ctarget-feature=-crt-static -C unsafe-allow-abi-mismatch=sanitizer

#![crate_type = "lib"]

pub fn foo1(_: fn(i32) -> i32) {}
// CHECK: define{{.*}}4foo1{{.*}}!type ![[TYPE1:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo2(_: fn(i32) -> i32, _: fn(i32) -> i32) {}
// CHECK: define{{.*}}4foo2{{.*}}!type ![[TYPE2:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo3(_: fn(i32) -> i32, _: fn(i32) -> i32, _: fn(i32) -> i32) {}
// CHECK: define{{.*}}4foo3{{.*}}!type ![[TYPE3:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo4(_: &dyn Fn(i32) -> i32) {}
// CHECK: define{{.*}}4foo4{{.*}}!type ![[TYPE4:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo5(_: &dyn Fn(i32) -> i32, _: &dyn Fn(i32) -> i32) {}
// CHECK: define{{.*}}4foo5{{.*}}!type ![[TYPE5:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo6(_: &dyn Fn(i32) -> i32, _: &dyn Fn(i32) -> i32, _: &dyn Fn(i32) -> i32) {}
// CHECK: define{{.*}}4foo6{{.*}}!type ![[TYPE6:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo7(_: &dyn FnMut(i32) -> i32) {}
// CHECK: define{{.*}}4foo7{{.*}}!type ![[TYPE7:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo8(_: &dyn FnMut(i32) -> i32, _: &dyn FnMut(i32) -> i32) {}
// CHECK: define{{.*}}4foo8{{.*}}!type ![[TYPE8:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo9(_: &dyn FnMut(i32) -> i32, _: &dyn FnMut(i32) -> i32, _: &dyn FnMut(i32) -> i32) {}
// CHECK: define{{.*}}4foo9{{.*}}!type ![[TYPE9:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo10(_: &dyn FnOnce(i32) -> i32) {}
// CHECK: define{{.*}}5foo10{{.*}}!type ![[TYPE10:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo11(_: &dyn FnOnce(i32) -> i32, _: &dyn FnOnce(i32) -> i32) {}
// CHECK: define{{.*}}5foo11{{.*}}!type ![[TYPE11:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo12(_: &dyn FnOnce(i32) -> i32, _: &dyn FnOnce(i32) -> i32, _: &dyn FnOnce(i32) -> i32) {}
// CHECK: define{{.*}}5foo12{{.*}}!type ![[TYPE12:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}

// CHECK: ![[TYPE1]] = !{i64 0, !"_ZTSFvPFu3i32S_EE"}
// CHECK: ![[TYPE2]] = !{i64 0, !"_ZTSFvPFu3i32S_ES0_E"}
// CHECK: ![[TYPE3]] = !{i64 0, !"_ZTSFvPFu3i32S_ES0_S0_E"}
// CHECK: ![[TYPE4]] = !{i64 0, !"_ZTSFvu3refIu3dynIu{{[0-9]+}}NtNtNtC{{[[:print:]]+}}_4core3ops8function2FnIu5tupleIu3i32EEu{{[0-9]+}}NtNtNtNtC{{[[:print:]]+}}_4core3ops8function6FnOnce6OutputIS0_ES_u6regionEEE"}
// CHECK: ![[TYPE5]] = !{i64 0, !"_ZTSFvu3refIu3dynIu{{[0-9]+}}NtNtNtC{{[[:print:]]+}}_4core3ops8function2FnIu5tupleIu3i32EEu{{[0-9]+}}NtNtNtNtC{{[[:print:]]+}}_4core3ops8function6FnOnce6OutputIS0_ES_u6regionEES5_E"}
// CHECK: ![[TYPE6]] = !{i64 0, !"_ZTSFvu3refIu3dynIu{{[0-9]+}}NtNtNtC{{[[:print:]]+}}_4core3ops8function2FnIu5tupleIu3i32EEu{{[0-9]+}}NtNtNtNtC{{[[:print:]]+}}_4core3ops8function6FnOnce6OutputIS0_ES_u6regionEES5_S5_E"}
// CHECK: ![[TYPE7]] = !{i64 0, !"_ZTSFvu3refIu3dynIu{{[0-9]+}}NtNtNtC{{[[:print:]]+}}_4core3ops8function5FnMutIu5tupleIu3i32EEu{{[0-9]+}}NtNtNtNtC{{[[:print:]]+}}_4core3ops8function6FnOnce6OutputIS0_ES_u6regionEEE"}
// CHECK: ![[TYPE8]] = !{i64 0, !"_ZTSFvu3refIu3dynIu{{[0-9]+}}NtNtNtC{{[[:print:]]+}}_4core3ops8function5FnMutIu5tupleIu3i32EEu{{[0-9]+}}NtNtNtNtC{{[[:print:]]+}}_4core3ops8function6FnOnce6OutputIS0_ES_u6regionEES5_E"}
// CHECK: ![[TYPE9]] = !{i64 0, !"_ZTSFvu3refIu3dynIu{{[0-9]+}}NtNtNtC{{[[:print:]]+}}_4core3ops8function5FnMutIu5tupleIu3i32EEu{{[0-9]+}}NtNtNtNtC{{[[:print:]]+}}_4core3ops8function6FnOnce6OutputIS0_ES_u6regionEES5_S5_E"}
// CHECK: ![[TYPE10]] = !{i64 0, !"_ZTSFvu3refIu3dynIu{{[0-9]+}}NtNtNtC{{[[:print:]]+}}_4core3ops8function6FnOnceIu5tupleIu3i32EEu{{[0-9]+}}NtNtNtNtC{{[[:print:]]+}}_4core3ops8function6FnOnce6OutputIS0_ES_u6regionEEE"}
// CHECK: ![[TYPE11]] = !{i64 0, !"_ZTSFvu3refIu3dynIu{{[0-9]+}}NtNtNtC{{[[:print:]]+}}_4core3ops8function6FnOnceIu5tupleIu3i32EEu{{[0-9]+}}NtNtNtNtC{{[[:print:]]+}}_4core3ops8function6FnOnce6OutputIS0_ES_u6regionEES5_E"}
// CHECK: ![[TYPE12]] = !{i64 0, !"_ZTSFvu3refIu3dynIu{{[0-9]+}}NtNtNtC{{[[:print:]]+}}_4core3ops8function6FnOnceIu5tupleIu3i32EEu{{[0-9]+}}NtNtNtNtC{{[[:print:]]+}}_4core3ops8function6FnOnce6OutputIS0_ES_u6regionEES5_S5_E"}
