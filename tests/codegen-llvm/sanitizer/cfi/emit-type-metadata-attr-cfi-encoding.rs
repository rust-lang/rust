// Verifies that user-defined CFI encoding for types are emitted.
//
//@ needs-sanitizer-cfi
//@ compile-flags: -Clto -Cno-prepopulate-passes -Ctarget-feature=-crt-static -Zsanitizer=cfi -Copt-level=0 -C unsafe-allow-abi-mismatch=sanitizer

#![crate_type = "lib"]
#![feature(cfi_encoding, extern_types)]

#[cfi_encoding = "3Foo"]
pub struct Type1(i32);

extern "C" {
    #[cfi_encoding = "3Bar"]
    type Type2;
}

#[cfi_encoding = "3Baz"]
#[repr(transparent)]
pub struct Type3(i32);

#[cfi_encoding = "i"]
pub struct Type4(i32);

#[cfi_encoding = "j"]
#[repr(transparent)]
pub struct Type5(u32);

pub fn foo0(_: Type1) {}
// CHECK: define{{.*}}foo0{{.*}}!type ![[TYPE0:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo1(_: Type1, _: Type1) {}
// CHECK: define{{.*}}foo1{{.*}}!type ![[TYPE1:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo2(_: Type1, _: Type1, _: Type1) {}
// CHECK: define{{.*}}foo2{{.*}}!type ![[TYPE2:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo3(_: *mut Type2) {}
// CHECK: define{{.*}}foo3{{.*}}!type ![[TYPE3:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo4(_: *mut Type2, _: *mut Type2) {}
// CHECK: define{{.*}}foo4{{.*}}!type ![[TYPE4:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo5(_: *mut Type2, _: *mut Type2, _: *mut Type2) {}
// CHECK: define{{.*}}foo5{{.*}}!type ![[TYPE5:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo6(_: *mut Type3) {}
// CHECK: define{{.*}}foo6{{.*}}!type ![[TYPE6:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo7(_: *mut Type3, _: *mut Type3) {}
// CHECK: define{{.*}}foo7{{.*}}!type ![[TYPE7:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo8(_: *mut Type3, _: *mut Type3, _: *mut Type3) {}
// CHECK: define{{.*}}foo8{{.*}}!type ![[TYPE8:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo9(_: Type4) {}
// CHECK: define{{.*}}foo9{{.*}}!type ![[TYPE9:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo10(_: Type4, _: Type4) {}
// CHECK: define{{.*}}foo10{{.*}}!type ![[TYPE10:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo11(_: Type4, _: Type4, _: Type4) {}
// CHECK: define{{.*}}foo11{{.*}}!type ![[TYPE11:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo12(_: Type5) {}
// CHECK: define{{.*}}foo12{{.*}}!type ![[TYPE12:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo13(_: Type5, _: Type5) {}
// CHECK: define{{.*}}foo13{{.*}}!type ![[TYPE13:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo14(_: Type5, _: Type5, _: Type5) {}
// CHECK: define{{.*}}foo14{{.*}}!type ![[TYPE14:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}

// CHECK: ![[TYPE0]] = !{i64 0, !"_ZTSFv3FooE"}
// CHECK: ![[TYPE1]] = !{i64 0, !"_ZTSFv3FooS_E"}
// CHECK: ![[TYPE2]] = !{i64 0, !"_ZTSFv3FooS_S_E"}
// CHECK: ![[TYPE3]] = !{i64 0, !"_ZTSFvP3BarE"}
// CHECK: ![[TYPE4]] = !{i64 0, !"_ZTSFvP3BarS0_E"}
// CHECK: ![[TYPE5]] = !{i64 0, !"_ZTSFvP3BarS0_S0_E"}
// CHECK: ![[TYPE6]] = !{i64 0, !"_ZTSFvP3BazE"}
// CHECK: ![[TYPE7]] = !{i64 0, !"_ZTSFvP3BazS0_E"}
// CHECK: ![[TYPE8]] = !{i64 0, !"_ZTSFvP3BazS0_S0_E"}
// CHECK: ![[TYPE9]] = !{i64 0, !"_ZTSFviE"}
// CHECK: ![[TYPE10]] = !{i64 0, !"_ZTSFviiE"}
// CHECK: ![[TYPE11]] = !{i64 0, !"_ZTSFviiiE"}
// CHECK: ![[TYPE12]] = !{i64 0, !"_ZTSFvjE"}
// CHECK: ![[TYPE13]] = !{i64 0, !"_ZTSFvjjE"}
// CHECK: ![[TYPE14]] = !{i64 0, !"_ZTSFvjjjE"}
