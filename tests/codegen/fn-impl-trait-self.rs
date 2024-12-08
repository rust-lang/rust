//@ compile-flags: -g
//@ ignore-wasi wasi codegens the main symbol differently
//
// CHECK-LABEL: @main
// MSVC: {{.*}}DIDerivedType(tag: DW_TAG_pointer_type, name: "recursive_type$ (*)()",{{.*}}
// NONMSVC: {{.*}}DIDerivedType(tag: DW_TAG_pointer_type, name: "fn() -> <recursive_type>",{{.*}}
//
// CHECK: {{.*}}DISubroutineType{{.*}}
// CHECK: {{.*}}DIBasicType(name: "<recur_type>", size: {{32|64}}, encoding: DW_ATE_unsigned)

pub fn foo() -> impl Copy {
    foo
}

fn main() {
    let my_res = foo();
}
