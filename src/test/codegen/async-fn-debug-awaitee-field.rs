// This test makes sure that the generator field capturing the awaitee in a `.await` expression
// is called "__awaitee" in debuginfo. This name must not be changed since debuggers and debugger
// extensions rely on the field having this name.

// ignore-tidy-linelength
// compile-flags: -C debuginfo=2 --edition=2018

async fn foo() {}

async fn async_fn_test() {
    foo().await;
}

// NONMSVC: [[GEN:!.*]] = !DICompositeType(tag: DW_TAG_structure_type, name: "{async_fn_env#0}",
// MSVC: [[GEN:!.*]] = !DICompositeType(tag: DW_TAG_union_type, name: "enum2$<async_fn_debug_awaitee_field::async_fn_test::async_fn_env$0>",
// CHECK: [[SUSPEND_STRUCT:!.*]] = !DICompositeType(tag: DW_TAG_structure_type, name: "Suspend0", scope: [[GEN]],
// CHECK: !DIDerivedType(tag: DW_TAG_member, name: "__awaitee", scope: [[SUSPEND_STRUCT]], {{.*}}, baseType: [[AWAITEE_TYPE:![0-9]*]],
// NONMSVC: [[AWAITEE_TYPE]] = !DICompositeType(tag: DW_TAG_structure_type, name: "GenFuture<async_fn_debug_awaitee_field::foo::{async_fn_env#0}>",
// MSVC: [[AWAITEE_TYPE]] = !DICompositeType(tag: DW_TAG_structure_type, name: "GenFuture<enum2$<async_fn_debug_awaitee_field::foo::async_fn_env$0> >",

fn main() {
    let _fn = async_fn_test();
}
