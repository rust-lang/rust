// compile-flags: -g
//
// CHECK-LABEL: @main
// CHECK: {{.*}}DIDerivedType(tag: DW_TAG_pointer_type, name: "fn() -> <recursive_type>",{{.*}}
//
// CHECK: {{.*}}DISubroutineType{{.*}}
// CHECK: {{.*}}DIBasicType(name: "<recur_type>", encoding: DW_ATE_unsigned)

pub fn foo() -> impl Copy {
    foo
}

fn main() {
    let my_res = foo();
}
