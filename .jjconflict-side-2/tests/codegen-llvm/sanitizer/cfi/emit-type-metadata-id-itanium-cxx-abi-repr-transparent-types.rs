// Verifies that type metadata identifiers for functions are emitted correctly
// for repr transparent types.
//
//@ needs-sanitizer-cfi
//@ compile-flags: -Clto -Cno-prepopulate-passes -Copt-level=0 -Zsanitizer=cfi -Ctarget-feature=-crt-static -C unsafe-allow-abi-mismatch=sanitizer

#![crate_type = "lib"]

extern crate core;
use core::ffi::*;
use std::marker::PhantomData;

struct Foo(i32);

// repr(transparent) user-defined type
#[repr(transparent)]
pub struct Type1 {
    member1: (),
    member2: PhantomData<i32>,
    member3: Foo,
}

// Self-referencing repr(transparent) user-defined type
#[repr(transparent)]
pub struct Type2<'a> {
    member1: (),
    member2: PhantomData<i32>,
    member3: &'a Type2<'a>,
}

pub struct Bar(i32);

// repr(transparent) user-defined generic type
#[repr(transparent)]
pub struct Type3<T>(T);

// repr(transparent) wrapper which engages in self-reference
#[repr(transparent)]
pub struct Type4(Type4Helper<Type4>);
#[repr(transparent)]
pub struct Type4Helper<T>(*mut T);

pub fn foo1(_: Type1) {}
// CHECK: define{{.*}}4foo1{{.*}}!type ![[TYPE1:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo2(_: Type1, _: Type1) {}
// CHECK: define{{.*}}4foo2{{.*}}!type ![[TYPE2:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo3(_: Type1, _: Type1, _: Type1) {}
// CHECK: define{{.*}}4foo3{{.*}}!type ![[TYPE3:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo4(_: Type2) {}
// CHECK: define{{.*}}4foo4{{.*}}!type ![[TYPE4:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo5(_: Type2, _: Type2) {}
// CHECK: define{{.*}}4foo5{{.*}}!type ![[TYPE5:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo6(_: Type2, _: Type2, _: Type2) {}
// CHECK: define{{.*}}4foo6{{.*}}!type ![[TYPE6:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo7(_: Type3<Bar>) {}
// CHECK: define{{.*}}4foo7{{.*}}!type ![[TYPE7:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo8(_: Type3<Bar>, _: Type3<Bar>) {}
// CHECK: define{{.*}}4foo8{{.*}}!type ![[TYPE8:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo9(_: Type3<Bar>, _: Type3<Bar>, _: Type3<Bar>) {}
// CHECK: define{{.*}}4foo9{{.*}}!type ![[TYPE9:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo10(_: Type4) {}
// CHECK: define{{.*}}5foo10{{.*}}!type ![[TYPE10:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo11(_: Type4, _: Type4) {}
// CHECK: define{{.*}}5foo11{{.*}}!type ![[TYPE11:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo12(_: Type4, _: Type4, _: Type4) {}
// CHECK: define{{.*}}5foo12{{.*}}!type ![[TYPE12:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}

// CHECK: ![[TYPE1]] = !{i64 0, !"_ZTSFvu{{[0-9]+}}NtC{{[[:print:]]+}}_{{[[:print:]]+}}3FooE"}
// CHECK: ![[TYPE2]] = !{i64 0, !"_ZTSFvu{{[0-9]+}}NtC{{[[:print:]]+}}_{{[[:print:]]+}}3FooS_E"}
// CHECK: ![[TYPE3]] = !{i64 0, !"_ZTSFvu{{[0-9]+}}NtC{{[[:print:]]+}}_{{[[:print:]]+}}3FooS_S_E"}
// CHECK: ![[TYPE4]] = !{i64 0, !"_ZTSFvu3refIvEE"}
// CHECK: ![[TYPE5]] = !{i64 0, !"_ZTSFvu3refIvES_E"}
// CHECK: ![[TYPE6]] = !{i64 0, !"_ZTSFvu3refIvES_S_E"}
// CHECK: ![[TYPE7]] = !{i64 0, !"_ZTSFvu{{[0-9]+}}NtC{{[[:print:]]+}}_{{[[:print:]]+}}3BarE"}
// CHECK: ![[TYPE8]] = !{i64 0, !"_ZTSFvu{{[0-9]+}}NtC{{[[:print:]]+}}_{{[[:print:]]+}}3BarS_E"}
// CHECK: ![[TYPE9]] = !{i64 0, !"_ZTSFvu{{[0-9]+}}NtC{{[[:print:]]+}}_{{[[:print:]]+}}3BarS_S_E"}
// CHECK: ![[TYPE10]] = !{i64 0, !"_ZTSFvPu{{[0-9]+}}NtC{{[[:print:]]+}}_{{[[:print:]]+}}5Type4E"}
// CHECK: ![[TYPE11]] = !{i64 0, !"_ZTSFvPu{{[0-9]+}}NtC{{[[:print:]]+}}_{{[[:print:]]+}}5Type4S0_E"}
// CHECK: ![[TYPE12]] = !{i64 0, !"_ZTSFvPu{{[0-9]+}}NtC{{[[:print:]]+}}_{{[[:print:]]+}}5Type4S0_S0_E"}
