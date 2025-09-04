// Verifies that integer types are normalized.
//
//@ needs-sanitizer-cfi
//@ compile-flags: -Clto -Cno-prepopulate-passes -Ctarget-feature=-crt-static -Zsanitizer=cfi -Zsanitizer-cfi-normalize-integers -Copt-level=0 -C unsafe-allow-abi-mismatch=sanitizer,sanitizer-cfi-normalize-integers

#![crate_type = "lib"]

extern crate core;

pub fn foo0(_: bool) {}
// CHECK: define{{.*}}foo0{{.*}}!type ![[TYPE0:[0-9]+]] !type !{{[0-9]+}}
pub fn foo1(_: bool, _: bool) {}
// CHECK: define{{.*}}foo1{{.*}}!type ![[TYPE1:[0-9]+]] !type !{{[0-9]+}}
pub fn foo2(_: bool, _: bool, _: bool) {}
// CHECK: define{{.*}}foo2{{.*}}!type ![[TYPE2:[0-9]+]] !type !{{[0-9]+}}
pub fn foo3(_: char) {}
// CHECK: define{{.*}}foo3{{.*}}!type ![[TYPE3:[0-9]+]] !type !{{[0-9]+}}
pub fn foo4(_: char, _: char) {}
// CHECK: define{{.*}}foo4{{.*}}!type ![[TYPE4:[0-9]+]] !type !{{[0-9]+}}
pub fn foo5(_: char, _: char, _: char) {}
// CHECK: define{{.*}}foo5{{.*}}!type ![[TYPE5:[0-9]+]] !type !{{[0-9]+}}
pub fn foo6(_: isize) {}
// CHECK: define{{.*}}foo6{{.*}}!type ![[TYPE6:[0-9]+]] !type !{{[0-9]+}}
pub fn foo7(_: isize, _: isize) {}
// CHECK: define{{.*}}foo7{{.*}}!type ![[TYPE7:[0-9]+]] !type !{{[0-9]+}}
pub fn foo8(_: isize, _: isize, _: isize) {}
// CHECK: define{{.*}}foo8{{.*}}!type ![[TYPE8:[0-9]+]] !type !{{[0-9]+}}
pub fn foo9(_: (), _: usize) {}
// CHECK: define{{.*}}foo9{{.*}}!type ![[TYPE9:[0-9]+]] !type !{{[0-9]+}}
pub fn foo10(_: (), _: usize, _: usize) {}
// CHECK: define{{.*}}foo10{{.*}}!type ![[TYPE10:[0-9]+]] !type !{{[0-9]+}}
pub fn foo11(_: (), _: usize, _: usize, _: usize) {}
// CHECK: define{{.*}}foo11{{.*}}!type ![[TYPE11:[0-9]+]] !type !{{[0-9]+}}

// CHECK: ![[TYPE0]] = !{i64 0, !"_ZTSFvu2u8E.normalized"}
// CHECK: ![[TYPE1]] = !{i64 0, !"_ZTSFvu2u8S_E.normalized"}
// CHECK: ![[TYPE2]] = !{i64 0, !"_ZTSFvu2u8S_S_E.normalized"}
// CHECK: ![[TYPE3]] = !{i64 0, !"_ZTSFvu3u32E.normalized"}
// CHECK: ![[TYPE4]] = !{i64 0, !"_ZTSFvu3u32S_E.normalized"}
// CHECK: ![[TYPE5]] = !{i64 0, !"_ZTSFvu3u32S_S_E.normalized"}
// CHECK: ![[TYPE6]] = !{i64 0, !"_ZTSFv{{u3i16|u3i32|u3i64|u4i128}}E.normalized"}
// CHECK: ![[TYPE7]] = !{i64 0, !"_ZTSFv{{u3i16|u3i32|u3i64|u4i128}}S_E.normalized"}
// CHECK: ![[TYPE8]] = !{i64 0, !"_ZTSFv{{u3i16|u3i32|u3i64|u4i128}}S_S_E.normalized"}
// CHECK: ![[TYPE9]] = !{i64 0, !"_ZTSFv{{u3u16|u3u32|u3u64|u4u128}}E.normalized"}
// CHECK: ![[TYPE10]] = !{i64 0, !"_ZTSFv{{u3u16|u3u32|u3u64|u4u128}}S_E.normalized"}
// CHECK: ![[TYPE11]] = !{i64 0, !"_ZTSFv{{u3u16|u3u32|u3u64|u4u128}}S_S_E.normalized"}
