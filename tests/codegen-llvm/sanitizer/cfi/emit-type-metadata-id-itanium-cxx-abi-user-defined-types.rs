// Verifies that type metadata identifiers for functions are emitted correctly
// for user-defined types.
//
//@ needs-sanitizer-cfi
//@ compile-flags: -Clto -Cno-prepopulate-passes -Copt-level=0 -Zsanitizer=cfi -Ctarget-feature=-crt-static -C unsafe-allow-abi-mismatch=sanitizer

#![crate_type = "lib"]
#![feature(extern_types)]

pub struct Struct1<T> {
    member1: T,
}

pub enum Enum1<T> {
    Variant1(T),
}

pub union Union1<T> {
    member1: std::mem::ManuallyDrop<T>,
}

extern "C" {
    pub type type1;
}

pub fn foo1(_: &Struct1<i32>) {}
// CHECK: define{{.*}}4foo1{{.*}}!type ![[TYPE1:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo2(_: &Struct1<i32>, _: &Struct1<i32>) {}
// CHECK: define{{.*}}4foo2{{.*}}!type ![[TYPE2:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo3(_: &Struct1<i32>, _: &Struct1<i32>, _: &Struct1<i32>) {}
// CHECK: define{{.*}}4foo3{{.*}}!type ![[TYPE3:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo4(_: &Enum1<i32>) {}
// CHECK: define{{.*}}4foo4{{.*}}!type ![[TYPE4:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo5(_: &Enum1<i32>, _: &Enum1<i32>) {}
// CHECK: define{{.*}}4foo5{{.*}}!type ![[TYPE5:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo6(_: &Enum1<i32>, _: &Enum1<i32>, _: &Enum1<i32>) {}
// CHECK: define{{.*}}4foo6{{.*}}!type ![[TYPE6:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo7(_: &Union1<i32>) {}
// CHECK: define{{.*}}4foo7{{.*}}!type ![[TYPE7:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo8(_: &Union1<i32>, _: &Union1<i32>) {}
// CHECK: define{{.*}}4foo8{{.*}}!type ![[TYPE8:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo9(_: &Union1<i32>, _: &Union1<i32>, _: &Union1<i32>) {}
// CHECK: define{{.*}}4foo9{{.*}}!type ![[TYPE9:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo10(_: *mut type1) {}
// CHECK: define{{.*}}5foo10{{.*}}!type ![[TYPE10:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo11(_: *mut type1, _: *mut type1) {}
// CHECK: define{{.*}}5foo11{{.*}}!type ![[TYPE11:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo12(_: *mut type1, _: *mut type1, _: *mut type1) {}
// CHECK: define{{.*}}5foo12{{.*}}!type ![[TYPE12:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}

// CHECK: ![[TYPE1]] = !{i64 0, !"_ZTSFvu3refIu{{[0-9]+}}NtC{{[[:print:]]+}}_{{[[:print:]]+}}7Struct1Iu3i32EEE"}
// CHECK: ![[TYPE2]] = !{i64 0, !"_ZTSFvu3refIu{{[0-9]+}}NtC{{[[:print:]]+}}_{{[[:print:]]+}}7Struct1Iu3i32EES1_E"}
// CHECK: ![[TYPE3]] = !{i64 0, !"_ZTSFvu3refIu{{[0-9]+}}NtC{{[[:print:]]+}}_{{[[:print:]]+}}7Struct1Iu3i32EES1_S1_E"}
// CHECK: ![[TYPE4]] = !{i64 0, !"_ZTSFvu3refIu{{[0-9]+}}NtC{{[[:print:]]+}}_{{[[:print:]]+}}5Enum1Iu3i32EEE"}
// CHECK: ![[TYPE5]] = !{i64 0, !"_ZTSFvu3refIu{{[0-9]+}}NtC{{[[:print:]]+}}_{{[[:print:]]+}}5Enum1Iu3i32EES1_E"}
// CHECK: ![[TYPE6]] = !{i64 0, !"_ZTSFvu3refIu{{[0-9]+}}NtC{{[[:print:]]+}}_{{[[:print:]]+}}5Enum1Iu3i32EES1_S1_E"}
// CHECK: ![[TYPE7]] = !{i64 0, !"_ZTSFvu3refIu{{[0-9]+}}NtC{{[[:print:]]+}}_{{[[:print:]]+}}6Union1Iu3i32EEE"}
// CHECK: ![[TYPE8]] = !{i64 0, !"_ZTSFvu3refIu{{[0-9]+}}NtC{{[[:print:]]+}}_{{[[:print:]]+}}6Union1Iu3i32EES1_E"}
// CHECK: ![[TYPE9]] = !{i64 0, !"_ZTSFvu3refIu{{[0-9]+}}NtC{{[[:print:]]+}}_{{[[:print:]]+}}6Union1Iu3i32EES1_S1_E"}
// CHECK: ![[TYPE10]] = !{i64 0, !"_ZTSFvP5type1E"}
// CHECK: ![[TYPE11]] = !{i64 0, !"_ZTSFvP5type1S0_E"}
// CHECK: ![[TYPE12]] = !{i64 0, !"_ZTSFvP5type1S0_S0_E"}
