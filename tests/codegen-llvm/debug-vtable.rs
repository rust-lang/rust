// ignore-tidy-linelength
//! This test checks the debuginfo for the expected 3 vtables is generated for correct names and
//! number of entries.

//@ revisions: MSVC NONMSVC
//@[MSVC] only-msvc
//@[NONMSVC] ignore-msvc

// Use the v0 symbol mangling scheme to codegen order independent of rustc version.
// Unnamed items like shims are generated in lexicographical order of their symbol name and in the
// legacy mangling scheme rustc version and generic parameters are both hashed into a single part
// of the name, thus randomizing item order with respect to rustc version.

//@ compile-flags: -Cdebuginfo=2 -Copt-level=0 -Csymbol-mangling-version=v0

// Make sure that vtables don't have the unnamed_addr attribute when debuginfo is enabled.
// This helps debuggers more reliably map from dyn pointer to concrete type.
// CHECK: @vtable.2 = private constant [
// CHECK: @vtable.3 = private constant <{
// CHECK: @vtable.4 = private constant <{

// NONMSVC: ![[USIZE:[0-9]+]] = !DIBasicType(name: "usize"
// MSVC: ![[USIZE:[0-9]+]] = !DIDerivedType(tag: DW_TAG_typedef, name: "usize"
// NONMSVC: ![[PTR:[0-9]+]] = !DIDerivedType(tag: DW_TAG_pointer_type, name: "*const ()"
// MSVC: ![[PTR:[0-9]+]] = !DIDerivedType(tag: DW_TAG_pointer_type, name: "ptr_const$<tuple$<> >"

// NONMSVC: !DIGlobalVariable(name: "<debug_vtable::Foo as debug_vtable::SomeTrait>::{vtable}"
// MSVC: !DIGlobalVariable(name: "impl$<debug_vtable::Foo, debug_vtable::SomeTrait>::vtable$"

// NONMSVC: ![[VTABLE_TY0:[0-9]+]] = !DICompositeType(tag: DW_TAG_structure_type, name: "<debug_vtable::Foo as debug_vtable::SomeTrait>::{vtable_type}", {{.*}} size: {{320|160}}, align: {{64|32}}, flags: DIFlagArtificial, {{.*}} vtableHolder: ![[FOO_TYPE:[0-9]+]],
// MSVC: ![[VTABLE_TY0:[0-9]+]] = !DICompositeType(tag: DW_TAG_structure_type, name: "impl$<debug_vtable::Foo, debug_vtable::SomeTrait>::vtable_type$", {{.*}} size: {{320|160}}, align: {{64|32}}, flags: DIFlagArtificial, {{.*}} vtableHolder: ![[FOO_TYPE:[0-9]+]],
// CHECK: !DIDerivedType(tag: DW_TAG_member, name: "drop_in_place", scope: ![[VTABLE_TY0]], {{.*}} baseType: ![[PTR]], size: {{64|32}}, align: {{64|32}})
// CHECK: !DIDerivedType(tag: DW_TAG_member, name: "size", scope: ![[VTABLE_TY0]], {{.*}} baseType: ![[USIZE]], size: {{64|32}}, align: {{64|32}}, offset: {{64|32}})
// CHECK: !DIDerivedType(tag: DW_TAG_member, name: "align", scope: ![[VTABLE_TY0]], {{.*}} baseType: ![[USIZE]], size: {{64|32}}, align: {{64|32}}, offset: {{128|64}})
// CHECK: !DIDerivedType(tag: DW_TAG_member, name: "__method3", scope: ![[VTABLE_TY0]], {{.*}} baseType: ![[PTR]], size: {{64|32}}, align: {{64|32}}, offset: {{192|96}})
// CHECK: !DIDerivedType(tag: DW_TAG_member, name: "__method4", scope: ![[VTABLE_TY0]], {{.*}} baseType: ![[PTR]], size: {{64|32}}, align: {{64|32}}, offset: {{256|128}})
// CHECK: ![[FOO_TYPE]] = !DICompositeType(tag: DW_TAG_structure_type, name: "Foo",

// NONMSVC: !DIGlobalVariable(name: "<debug_vtable::Foo as debug_vtable::SomeTraitWithGenerics<u64, i8>>::{vtable}"
// MSVC: !DIGlobalVariable(name: "impl$<debug_vtable::Foo, debug_vtable::SomeTraitWithGenerics<u64,i8> >::vtable$"

// NONMSVC: ![[VTABLE_TY1:[0-9]+]] = !DICompositeType(tag: DW_TAG_structure_type, name: "<debug_vtable::Foo as debug_vtable::SomeTraitWithGenerics<u64, i8>>::{vtable_type}", {{.*}}, size: {{256|128}}, align: {{64|32}}, flags: DIFlagArtificial, {{.*}}, vtableHolder: ![[FOO_TYPE]],
// MSVC: ![[VTABLE_TY1:[0-9]+]] = !DICompositeType(tag: DW_TAG_structure_type, name: "impl$<debug_vtable::Foo, debug_vtable::SomeTraitWithGenerics<u64,i8> >::vtable_type$", {{.*}}, size: {{256|128}}, align: {{64|32}}, flags: DIFlagArtificial, {{.*}}, vtableHolder: ![[FOO_TYPE]],
// CHECK: !DIDerivedType(tag: DW_TAG_member, name: "drop_in_place", scope: ![[VTABLE_TY1]], {{.*}} baseType: ![[PTR]], size: {{64|32}}, align: {{64|32}})
// CHECK: !DIDerivedType(tag: DW_TAG_member, name: "size", scope: ![[VTABLE_TY1]], {{.*}} baseType: ![[USIZE]], size: {{64|32}}, align: {{64|32}}, offset: {{64|32}})
// CHECK: !DIDerivedType(tag: DW_TAG_member, name: "align", scope: ![[VTABLE_TY1]], {{.*}} baseType: ![[USIZE]], size: {{64|32}}, align: {{64|32}}, offset: {{128|64}})
// CHECK: !DIDerivedType(tag: DW_TAG_member, name: "__method3", scope: ![[VTABLE_TY1]], {{.*}} baseType: ![[PTR]], size: {{64|32}}, align: {{64|32}}, offset: {{192|96}})

// NONMSVC: !DIGlobalVariable(name: "<debug_vtable::Foo as _>::{vtable}"
// MSVC: !DIGlobalVariable(name: "impl$<debug_vtable::Foo, _>::vtable$"

// NONMSVC: ![[VTABLE_TY2:[0-9]+]] = !DICompositeType(tag: DW_TAG_structure_type, name: "<debug_vtable::Foo as _>::{vtable_type}", {{.*}}, size: {{192|96}}, align: {{64|32}}, flags: DIFlagArtificial, {{.*}}, vtableHolder: ![[FOO_TYPE]],
// MSVC: ![[VTABLE_TY2:[0-9]+]] = !DICompositeType(tag: DW_TAG_structure_type, name: "impl$<debug_vtable::Foo, _>::vtable_type$", {{.*}}, size: {{192|96}}, align: {{64|32}}, flags: DIFlagArtificial, {{.*}}, vtableHolder: ![[FOO_TYPE]],
// CHECK: !DIDerivedType(tag: DW_TAG_member, name: "drop_in_place", scope: ![[VTABLE_TY2]], {{.*}}, baseType: ![[PTR]], size: {{64|32}}, align: {{64|32}})
// CHECK: !DIDerivedType(tag: DW_TAG_member, name: "size", scope: ![[VTABLE_TY2]], {{.*}}, baseType: ![[USIZE]], size: {{64|32}}, align: {{64|32}}, offset: {{64|32}})
// CHECK: !DIDerivedType(tag: DW_TAG_member, name: "align", scope: ![[VTABLE_TY2]], {{.*}}, baseType: ![[USIZE]], size: {{64|32}}, align: {{64|32}}, offset: {{128|64}})

// NONMSVC: !DIGlobalVariable(name: "<debug_vtable::bar::{closure_env#0} as core::ops::function::FnOnce<(core::option::Option<&dyn core::ops::function::Fn<(), Output=()>>)>>::{vtable}"
// MSVC: !DIGlobalVariable(name: "impl$<debug_vtable::bar::closure_env$0, core::ops::function::FnOnce<tuple$<enum2$<core::option::Option<ref$<dyn$<core::ops::function::Fn<tuple$<>,assoc$<Output,tuple$<> > > > > > > > > >::vtable$"

// NONMSVC: !DIGlobalVariable(name: "<debug_vtable::generic_closure::{closure_env#0}<bool> as core::ops::function::FnOnce<()>>::{vtable}"
// MSVC: !DIGlobalVariable(name: "impl$<debug_vtable::generic_closure::closure_env$0<bool>, core::ops::function::FnOnce<tuple$<> > >::vtable$

// NONMSVC: !DIGlobalVariable(name: "<debug_vtable::generic_closure::{closure_env#0}<u32> as core::ops::function::FnOnce<()>>::{vtable}"
// MSVC: !DIGlobalVariable(name: "impl$<debug_vtable::generic_closure::closure_env$0<u32>, core::ops::function::FnOnce<tuple$<> > >::vtable$

#![crate_type = "lib"]

// Force emission for debuginfo for usize and *const() early..
pub static mut XYZ: Option<(usize, *const ())> = None;

pub struct Foo;

pub trait SomeTrait {
    fn method1(&self) -> u32;
    fn method2(&self) -> u32;
}

impl SomeTrait for Foo {
    fn method1(&self) -> u32 {
        1
    }
    fn method2(&self) -> u32 {
        2
    }
}

pub trait SomeTraitWithGenerics<T, U> {
    fn method1(&self) -> (T, U);
}

impl SomeTraitWithGenerics<u64, i8> for Foo {
    fn method1(&self) -> (u64, i8) {
        (1, 2)
    }
}

pub fn foo(x: &Foo) -> (u32, (u64, i8), &dyn Send) {
    let y: &dyn SomeTrait = x;
    let z: &dyn SomeTraitWithGenerics<u64, i8> = x;
    (y.method1(), z.method1(), x as &dyn Send)
}

// Constructing the debuginfo name for the FnOnce vtable below initially caused an ICE on MSVC
// because the trait type contains a late bound region that needed to be erased before the type
// layout for the niche enum `Option<&dyn Fn()>` could be computed.
pub fn bar() -> Box<dyn FnOnce(Option<&dyn Fn()>)> {
    Box::new(|_x: Option<&dyn Fn()>| {})
}

fn generic_closure<T: 'static>(x: T) -> Box<dyn FnOnce() -> T> {
    Box::new(move || x)
}

pub fn instantiate_generic_closures() -> (Box<dyn FnOnce() -> u32>, Box<dyn FnOnce() -> bool>) {
    (generic_closure(1u32), generic_closure(false))
}
