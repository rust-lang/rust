// compile-flags: -Cdebuginfo=2 -Copt-level=0 -Ccodegen-units=1
// ignore-tidy-linelength

// This test checks the debuginfo for the expected 3 vtables is generated for correct names and number
// of entries.

// NONMSVC-LABEL: !DIGlobalVariable(name: "<debug_vtable::Foo as debug_vtable::SomeTrait>::{vtable}"
// MSVC-LABEL: !DIGlobalVariable(name: "impl$<debug_vtable::Foo, debug_vtable::SomeTrait>::vtable$"
// NONMSVC: !DIDerivedType(tag: DW_TAG_pointer_type, name: "*const ()",
// MSVC: !DIDerivedType(tag: DW_TAG_pointer_type, name: "ptr_const$<tuple$<> >",
// CHECK: !DISubrange(count: 5

// NONMSVC-LABEL: !DIGlobalVariable(name: "<debug_vtable::Foo as debug_vtable::SomeTraitWithGenerics<u64, i8>>::{vtable}"
// MSVC-LABEL: !DIGlobalVariable(name: "impl$<debug_vtable::Foo, debug_vtable::SomeTraitWithGenerics<u64,i8> >::vtable$"
// CHECK: !DISubrange(count: 4

// NONMSVC-LABEL: !DIGlobalVariable(name: "<debug_vtable::Foo as _>::{vtable}"
// MSVC-LABEL: !DIGlobalVariable(name: "impl$<debug_vtable::Foo, _>::vtable$"
// CHECK: !DISubrange(count: 3

#![crate_type = "lib"]

pub struct Foo;

pub trait SomeTrait {
    fn method1(&self) -> u32;
    fn method2(&self) -> u32;
}

impl SomeTrait for Foo {
    fn method1(&self) -> u32 { 1 }
    fn method2(&self) -> u32 { 2 }
}

pub trait SomeTraitWithGenerics<T, U> {
    fn method1(&self) -> (T, U);
}

impl SomeTraitWithGenerics<u64, i8> for Foo {
    fn method1(&self) -> (u64, i8) { (1, 2) }
}

pub fn foo(x: &Foo) -> (u32, (u64, i8), &dyn Send) {
    let y: &dyn SomeTrait = x;
    let z: &dyn SomeTraitWithGenerics<u64, i8> = x;
    (y.method1(), z.method1(), x as &dyn Send)
}
