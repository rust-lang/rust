// This test checks the debuginfo for the expected 3 vtables is generated for correct names and number
// of entries.

// Use the v0 symbol mangling scheme to codegen order independent of rustc version.
// Unnamed items like shims are generated in lexicographical order of their symbol name and in the
// legacy mangling scheme rustc version and generic parameters are both hashed into a single part
// of the name, thus randomizing item order with respect to rustc version.

// compile-flags: -Cdebuginfo=2 -Copt-level=0 -Csymbol-mangling-version=v0
// ignore-tidy-linelength

// NONMSVC: !DIGlobalVariable(name: "<debug_vtable::Foo as debug_vtable::SomeTrait>::{vtable}"
// MSVC: !DIGlobalVariable(name: "impl$<debug_vtable::Foo, debug_vtable::SomeTrait>::vtable$"
// NONMSVC: !DIDerivedType(tag: DW_TAG_pointer_type, name: "*const ()",
// MSVC: !DIDerivedType(tag: DW_TAG_pointer_type, name: "ptr_const$<tuple$<> >",
// CHECK: !DISubrange(count: 5

// NONMSVC: !DIGlobalVariable(name: "<debug_vtable::Foo as debug_vtable::SomeTraitWithGenerics<u64, i8>>::{vtable}"
// MSVC: !DIGlobalVariable(name: "impl$<debug_vtable::Foo, debug_vtable::SomeTraitWithGenerics<u64,i8> >::vtable$"
// CHECK: !DISubrange(count: 4

// NONMSVC: !DIGlobalVariable(name: "<debug_vtable::Foo as _>::{vtable}"
// MSVC: !DIGlobalVariable(name: "impl$<debug_vtable::Foo, _>::vtable$"
// CHECK: !DISubrange(count: 3

// NONMSVC: !DIGlobalVariable(name: "<debug_vtable::bar::{closure_env#0} as core::ops::function::FnOnce<(core::option::Option<&dyn core::ops::function::Fn<(), Output=()>>)>>::{vtable}"
// MSVC: !DIGlobalVariable(name: "impl$<debug_vtable::bar::closure_env$0, core::ops::function::FnOnce<tuple$<enum$<core::option::Option<ref$<dyn$<core::ops::function::Fn<tuple$<>,assoc$<Output,tuple$<> > > > > >, {{.*}}, {{.*}}, Some> > > >::vtable$"

// NONMSVC: !DIGlobalVariable(name: "<debug_vtable::generic_closure::{closure_env#0}<bool> as core::ops::function::FnOnce<()>>::{vtable}"
// MSVC: !DIGlobalVariable(name: "impl$<debug_vtable::generic_closure::closure_env$0<bool>, core::ops::function::FnOnce<tuple$<> > >::vtable$

// NONMSVC: !DIGlobalVariable(name: "<debug_vtable::generic_closure::{closure_env#0}<u32> as core::ops::function::FnOnce<()>>::{vtable}"
// MSVC: !DIGlobalVariable(name: "impl$<debug_vtable::generic_closure::closure_env$0<u32>, core::ops::function::FnOnce<tuple$<> > >::vtable$

#![crate_type = "lib"]

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
