//@ compile-flags: -Zvirtual-function-elimination -Clto=fat -Zverify-llvm-ir -Cno-prepopulate-passes -Copt-level=0
//@ no-prefer-dynamic
#![crate_type = "staticlib"]

// Generating a constant vtable takes a different codegen path. Make sure that it also sets the
// !type and !vcall_visibility attributes.

// CHECK: @vtable.{{[0-9]+}} = {{.*}}StructB{{.*}} !type ![[META_SIZED:[0-9]+]], !type ![[TRAIT:[0-9]+]], !vcall_visibility ![[VIS:[0-9]+]]
// CHECK: @vtable.{{[0-9]+}} = {{.*}}StructA{{.*}} !type ![[META_SIZED]], !type ![[TRAIT]], !vcall_visibility ![[VIS]]

// CHECK-DAG: ![[META_SIZED]] = !{i64 0, !"{{.*}}9MetaSized"}
// CHECK-DAG: ![[TRAIT]] = !{i64 0, !"{{.*}}5Trait"}
// CHECK-DAG: ![[VIS]] = !{i64 1}

trait Trait {
    fn method(&self) -> i32;
}

struct StructA;
struct StructB;

impl Trait for StructA {
    #[inline(never)]
    fn method(&self) -> i32 {
        3
    }
}

impl Trait for StructB {
    #[inline(never)]
    fn method(&self) -> i32 {
        4
    }
}

const CONST: &dyn Trait = &StructA;

#[unsafe(no_mangle)]
fn call(x: &dyn Trait) -> i32 {
    x.method()
}

#[unsafe(no_mangle)]
fn usage() -> i32 {
    let a = CONST;
    let b: &dyn Trait = &StructB;

    call(a) + call(b)
}
