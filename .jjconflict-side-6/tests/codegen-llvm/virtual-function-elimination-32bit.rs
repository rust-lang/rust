//@ compile-flags: -Zvirtual-function-elimination -Clto -Copt-level=3 -Csymbol-mangling-version=v0
//@ ignore-64bit

// CHECK: @vtable.0 = {{.*}}, !type ![[TYPE0:[0-9]+]], !vcall_visibility ![[VCALL_VIS0:[0-9]+]]

#![crate_type = "lib"]

trait T {
    // CHECK-LABEL: ; <virtual_function_elimination_32bit::S as virtual_function_elimination_32bit::T>::used
    fn used(&self) -> i32 {
        1
    }
    // CHECK-LABEL-NOT: {{.*}}::unused
    fn unused(&self) -> i32 {
        2
    }
}

#[derive(Copy, Clone)]
struct S;

impl T for S {}

fn taking_t(t: &dyn T) -> i32 {
    // CHECK: @llvm.type.checked.load({{.*}}, i32 12, metadata !"[[MANGLED_TYPE0:[0-9a-zA-Z_]+]]")
    t.used()
}

pub fn main() {
    let s = S;
    taking_t(&s);
}

// CHECK: ![[TYPE0]] = !{i32 0, !"[[MANGLED_TYPE0]]"}
// CHECK: ![[VCALL_VIS0]] = !{i64 2}
