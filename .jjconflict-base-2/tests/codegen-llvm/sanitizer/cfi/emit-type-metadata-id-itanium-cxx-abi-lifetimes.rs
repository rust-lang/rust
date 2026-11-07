// Verifies that type metadata identifiers for functions are emitted correctly
// for lifetimes/regions.
//
//@ needs-sanitizer-cfi
//@ compile-flags: -Clto -Cno-prepopulate-passes -Ctarget-feature=-crt-static -Zsanitizer=cfi -Copt-level=0 -C unsafe-allow-abi-mismatch=sanitizer

#![crate_type = "lib"]
#![feature(type_alias_impl_trait)]

extern crate core;

pub type Type1 = impl Send;

#[define_opaque(Type1)]
pub fn foo<'a>()
where
    Type1: 'static,
{
    pub struct Foo<'a>(&'a i32);
    pub struct Bar<'a, 'b>(&'a i32, &'b Foo<'b>);
    let _: Type1 = Bar;
}

pub fn foo1(_: Type1) {}
// CHECK: define{{.*}}4foo1{{.*}}!type ![[TYPE1:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo2(_: Type1, _: Type1) {}
// CHECK: define{{.*}}4foo2{{.*}}!type ![[TYPE2:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
pub fn foo3(_: Type1, _: Type1, _: Type1) {}
// CHECK: define{{.*}}4foo3{{.*}}!type ![[TYPE3:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
