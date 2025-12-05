// Verifies that type metadata identifiers for functions are emitted correctly
// for const generics.
//
//@ needs-sanitizer-cfi
//@ compile-flags: -Clto -Cno-prepopulate-passes -Ctarget-feature=-crt-static -Zsanitizer=cfi -Copt-level=0 -C unsafe-allow-abi-mismatch=sanitizer

#![crate_type = "lib"]
#![feature(type_alias_impl_trait)]

extern crate core;

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

// CHECK: ![[TYPE1]] = !{i64 0, !"_ZTSFvu{{[0-9]+}}NtNvC{{[[:print:]]+}}_{{[[:print:]]+}}3foo3FooIu3i32Lu5usize32EEE"}
// CHECK: ![[TYPE2]] = !{i64 0, !"_ZTSFvu{{[0-9]+}}NtNvC{{[[:print:]]+}}_{{[[:print:]]+}}3foo3FooIu3i32Lu5usize32EES2_E"}
// CHECK: ![[TYPE3]] = !{i64 0, !"_ZTSFvu{{[0-9]+}}NtNvC{{[[:print:]]+}}_{{[[:print:]]+}}3foo3FooIu3i32Lu5usize32EES2_S2_E"}
