// ignore-tidy-linelength
//! This test makes sure that the coroutine field capturing the awaitee in a `.await` expression
//! is called `__awaitee` in debuginfo. This name must not be changed since debuggers and debugger
//! extensions rely on the field having this name.

//@ revisions: MSVC NONMSVC
//@[MSVC] only-msvc
//@[NONMSVC] ignore-msvc

//@ compile-flags: -C debuginfo=2 -Copt-level=0
//@ edition: 2018

#![crate_type = "lib"]

pub async fn async_fn_test() {
    foo().await;
}

pub async fn foo() {}

// NONMSVC-DAG: [[AWAITEE_TYPE:![0-9]*]] = !DICompositeType(tag: DW_TAG_structure_type, name: "{async_fn_env#0}", scope: [[AWAITEE_SCOPE:![0-9]*]],
// MSVC-DAG: [[AWAITEE_TYPE:![0-9]*]] = !DICompositeType(tag: DW_TAG_union_type, name: "enum2$<async_fn_debug_awaitee_field::foo::async_fn_env$0>",
// NONMSVC-DAG: [[AWAITEE_SCOPE]] = !DINamespace(name: "foo",
// NONMSVC-DAG: [[GEN:!.*]] = !DICompositeType(tag: DW_TAG_structure_type, name: "{async_fn_env#0}", scope: [[GEN_SCOPE:![0-9]*]],
// MSVC-DAG: [[GEN:!.*]] = !DICompositeType(tag: DW_TAG_union_type, name: "enum2$<async_fn_debug_awaitee_field::async_fn_test::async_fn_env$0>",
// NONMSVC-DAG: [[GEN_SCOPE:!.*]] = !DINamespace(name: "async_fn_test",
// CHECK-DAG: [[SUSPEND_STRUCT:!.*]] = !DICompositeType(tag: DW_TAG_structure_type, name: "Suspend0", scope: [[GEN]],
// CHECK-DAG: !DIDerivedType(tag: DW_TAG_member, name: "__awaitee", scope: [[SUSPEND_STRUCT]], {{.*}}, baseType: [[AWAITEE_TYPE]],
