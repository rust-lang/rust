// This test checks that we get proper type names for closure environments and
// async-fn environments in debuginfo, especially making sure that generic arguments
// of the enclosing functions don't get lost.
//
// Unfortunately, the order that debuginfo gets emitted into LLVM IR becomes a bit hard
// to predict once async fns are involved.
//
// Note that the test does not check async-fns when targeting MSVC because debuginfo for
// those does not follow the enum-fallback encoding yet and thus is incomplete.

// ignore-tidy-linelength

// Use the v0 symbol mangling scheme to codegen order independent of rustc version.
// Unnamed items like shims are generated in lexicographical order of their symbol name and in the
// legacy mangling scheme rustc version and generic parameters are both hashed into a single part
// of the name, thus randomizing item order with respect to rustc version.

// compile-flags: -Cdebuginfo=2 --edition 2021 -Copt-level=0 -Csymbol-mangling-version=v0

// non_generic_closure()
// NONMSVC: !DICompositeType(tag: DW_TAG_structure_type, name: "{closure_env#0}", scope: ![[non_generic_closure_NAMESPACE:[0-9]+]],
// MSVC: !DICompositeType(tag: DW_TAG_structure_type, name: "closure_env$0", scope: ![[non_generic_closure_NAMESPACE:[0-9]+]],
// CHECK: ![[non_generic_closure_NAMESPACE]] = !DINamespace(name: "non_generic_closure"

// CHECK: ![[function_containing_closure_NAMESPACE:[0-9]+]] = !DINamespace(name: "function_containing_closure"
// CHECK: ![[generic_async_function_NAMESPACE:[0-9]+]] = !DINamespace(name: "generic_async_function"
// CHECK: ![[generic_async_block_NAMESPACE:[0-9]+]] = !DINamespace(name: "generic_async_block"

// function_containing_closure<u32>()
// NONMSVC: !DICompositeType(tag: DW_TAG_structure_type, name: "{closure_env#0}<u32>", scope: ![[function_containing_closure_NAMESPACE]]
// MSVC: !DICompositeType(tag: DW_TAG_structure_type, name: "closure_env$0<u32>", scope: ![[function_containing_closure_NAMESPACE]]

// generic_async_function<Foo>()
// NONMSVC: !DICompositeType(tag: DW_TAG_structure_type, name: "{async_fn_env#0}<debuginfo_generic_closure_env_names::Foo>", scope: ![[generic_async_function_NAMESPACE]]

// generic_async_function<u32>()
// NONMSVC: !DICompositeType(tag: DW_TAG_structure_type, name: "{async_fn_env#0}<u32>", scope: ![[generic_async_function_NAMESPACE]]

// generic_async_block<Foo>()
// NONMSVC: !DICompositeType(tag: DW_TAG_structure_type, name: "{async_block_env#0}<debuginfo_generic_closure_env_names::Foo>", scope: ![[generic_async_block_NAMESPACE]]

// generic_async_block<u32>()
// NONMSVC: !DICompositeType(tag: DW_TAG_structure_type, name: "{async_block_env#0}<u32>", scope: ![[generic_async_block_NAMESPACE]]

// function_containing_closure<Foo>()
// NONMSVC: !DICompositeType(tag: DW_TAG_structure_type, name: "{closure_env#0}<debuginfo_generic_closure_env_names::Foo>", scope: ![[function_containing_closure_NAMESPACE]]
// MSVC: !DICompositeType(tag: DW_TAG_structure_type, name: "closure_env$0<debuginfo_generic_closure_env_names::Foo>", scope: ![[function_containing_closure_NAMESPACE]]


#![crate_type = "lib"]
use std::future::Future;

pub struct Foo;

pub fn non_generic_closure(x: Foo) -> Box<dyn FnOnce() -> Foo> {
    return Box::new(move || x);
}

fn function_containing_closure<T: 'static>(x: T) -> impl FnOnce() -> T {
    // This static only exists to trigger generating the namespace debuginfo for
    // `function_containing_closure` at a predictable, early point, which makes
    // writing the FileCheck tests above simpler.
    static _X: u8 = 0;

    return move || x;
}

async fn generic_async_function<T: 'static>(x: T) -> T {
    static _X: u8 = 0; // Same as above
    x
}

fn generic_async_block<T: 'static>(x: T) -> impl Future<Output=T> {
    static _X: u8 = 0; // Same as above
    async move {
        x
    }
}

pub fn instantiate_generics() {
    let _closure_u32 = function_containing_closure(7u32);
    let _closure_foo = function_containing_closure(Foo);

    let _async_fn_u32 = generic_async_function(42u32);
    let _async_fn_foo = generic_async_function(Foo);

    let _async_block_u32 = generic_async_block(64u32);
    let _async_block_foo = generic_async_block(Foo);
}
