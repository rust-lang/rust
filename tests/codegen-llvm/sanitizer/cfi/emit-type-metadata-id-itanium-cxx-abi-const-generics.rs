// Verifies that type metadata identifiers for functions are emitted correctly
// for const generics.
//
//@ needs-sanitizer-cfi
//@ compile-flags: -Clto -Cno-prepopulate-passes -Ctarget-feature=-crt-static -Zsanitizer=cfi -Copt-level=0 -C unsafe-allow-abi-mismatch=sanitizer

#![crate_type = "lib"]
#![feature(adt_const_params)]
#![feature(type_alias_impl_trait)]
#![feature(unsized_const_params)]
#![allow(incomplete_features)]

extern crate core;

use std::marker::ConstParamTy;

pub type Type1 = impl Send;

#[define_opaque(Type1)]
pub fn foo()
where
    Type1: 'static,
{
    pub struct Foo<T, const N: usize>([T; N]);
    let _: Type1 = Foo([0; 32]);
}

pub fn foo1(_: Type1) {}
// CHECK: define{{.*}}4foo1{{.*}}!type ![[TYPE1:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo2(_: Type1, _: Type1) {}
// CHECK: define{{.*}}4foo2{{.*}}!type ![[TYPE2:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo3(_: Type1, _: Type1, _: Type1) {}
// CHECK: define{{.*}}4foo3{{.*}}!type ![[TYPE3:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}

#[derive(PartialEq, Eq, ConstParamTy)]
pub struct Struct1 {
    pub x: u16,
    pub y: u16,
}

#[derive(PartialEq, Eq, ConstParamTy)]
pub enum Enum1 {
    Variant1,
    Variant2(u8),
}

pub struct BoolHolder<const B: bool>;
pub struct IntHolder<const I: i32>;
pub struct CharHolder<const C: char>;
pub struct StrHolder<const S: &'static str>;
pub struct StructHolder<const S: Struct1>;
pub struct EnumHolder<const E: Enum1>;
pub struct ArrayHolder<const A: [u16; 2]>;
pub struct TupleHolder<const T: (u16, bool)>;

pub fn foo4(_: &BoolHolder<true>) {}
// CHECK: define{{.*}}4foo4{{.*}}!type ![[TYPE4:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo5(_: &IntHolder<-1>) {}
// CHECK: define{{.*}}4foo5{{.*}}!type ![[TYPE5:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo6(_: &CharHolder<'x'>) {}
// CHECK: define{{.*}}4foo6{{.*}}!type ![[TYPE6:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo7(_: &StrHolder<"hello">) {}
// CHECK: define{{.*}}4foo7{{.*}}!type ![[TYPE7:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo8(_: &StructHolder<{ Struct1 { x: 1, y: 2 } }>) {}
// CHECK: define{{.*}}4foo8{{.*}}!type ![[TYPE8:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo9(_: &EnumHolder<{ Enum1::Variant2(5) }>) {}
// CHECK: define{{.*}}4foo9{{.*}}!type ![[TYPE9:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo10(_: &ArrayHolder<{ [3, 4] }>) {}
// CHECK: define{{.*}}5foo10{{.*}}!type ![[TYPE10:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo11(_: &TupleHolder<{ (6, true) }>) {}
// CHECK: define{{.*}}5foo11{{.*}}!type ![[TYPE11:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}

// CHECK: ![[TYPE1]] = !{i64 0, !"_ZTSFvu{{[0-9]+}}NtNvC{{[[:print:]]+}}_{{[[:print:]]+}}3foo3FooIu3i32Lu5usize32EEE"}
// CHECK: ![[TYPE2]] = !{i64 0, !"_ZTSFvu{{[0-9]+}}NtNvC{{[[:print:]]+}}_{{[[:print:]]+}}3foo3FooIu3i32Lu5usize32EES2_E"}
// CHECK: ![[TYPE3]] = !{i64 0, !"_ZTSFvu{{[0-9]+}}NtNvC{{[[:print:]]+}}_{{[[:print:]]+}}3foo3FooIu3i32Lu5usize32EES2_S2_E"}
// CHECK: ![[TYPE4]] = !{i64 0, !"_ZTSFvu3refIu{{[0-9]+}}NtC{{[[:print:]]+}}_{{[[:print:]]+}}10BoolHolderILb1EEEE"}
// CHECK: ![[TYPE5]] = !{i64 0, !"_ZTSFvu3refIu{{[0-9]+}}NtC{{[[:print:]]+}}_{{[[:print:]]+}}9IntHolderILu3i32n1EEEE"}
// CHECK: ![[TYPE6]] = !{i64 0, !"_ZTSFvu3refIu{{[0-9]+}}NtC{{[[:print:]]+}}_{{[[:print:]]+}}10CharHolderILu4char120EEEE"}
// CHECK: ![[TYPE7]] = !{i64 0, !"_ZTSFvu3refIu{{[0-9]+}}NtC{{[[:print:]]+}}_{{[[:print:]]+}}9StrHolderILu3refIu3strE68656c6c6fEEEE"}
// CHECK: ![[TYPE8]] = !{i64 0, !"_ZTSFvu3refIu{{[0-9]+}}NtC{{[[:print:]]+}}_{{[[:print:]]+}}12StructHolderILu{{[0-9]+}}NtC{{[[:print:]]+}}_{{[[:print:]]+}}7Struct1Lu3u161ELS0_2EEEEE"}
// CHECK: ![[TYPE9]] = !{i64 0, !"_ZTSFvu3refIu{{[0-9]+}}NtC{{[[:print:]]+}}_{{[[:print:]]+}}10EnumHolderILu{{[0-9]+}}NtC{{[[:print:]]+}}_{{[[:print:]]+}}5Enum1V1Lu2u85EEEEE"}
// CHECK: ![[TYPE10]] = !{i64 0, !"_ZTSFvu3refIu{{[0-9]+}}NtC{{[[:print:]]+}}_{{[[:print:]]+}}11ArrayHolderILA2u3u16LS_3ELS_4EEEEE"}
// CHECK: ![[TYPE11]] = !{i64 0, !"_ZTSFvu3refIu{{[0-9]+}}NtC{{[[:print:]]+}}_{{[[:print:]]+}}11TupleHolderILu5tupleIu3u16bELS_6ELb1EEEEE"}
