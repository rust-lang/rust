// Just make sure that async closures don't ICE.
//
//@ compile-flags: -C debuginfo=2
//@ edition: 2018
//@ ignore-msvc

// CHECK-DAG:  [[GEN_FN:!.*]] = !DINamespace(name: "async_closure_test"
// CHECK-DAG:  [[CLOSURE:!.*]] = !DICompositeType(tag: DW_TAG_structure_type, name: "{closure_env#0}", scope: [[GEN_FN]]
// CHECK-DAG:  [[UPVAR:!.*]] = !DIDerivedType(tag: DW_TAG_member, name: "upvar", scope: [[CLOSURE]]

fn async_closure_test(upvar: &str) -> impl AsyncFn() + '_ {
    async move || {
        let hello = String::from("hello");
        println!("{hello}, {upvar}");
    }
}

fn main() {
    let _async_closure = async_closure_test("world");
}
