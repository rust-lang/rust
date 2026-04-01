//@ compile-flags: -Zvirtual-function-elimination -Clto -Copt-level=3 -Csymbol-mangling-version=v0
//@ ignore-32bit

// CHECK: @vtable.0 = {{.*}}, !type ![[TYPE0:[0-9]+]], !vcall_visibility ![[VCALL_VIS0:[0-9]+]]
// CHECK: @vtable.1 = {{.*}}, !type ![[TYPE1:[0-9]+]], !vcall_visibility ![[VCALL_VIS0:[0-9]+]]
// CHECK: @vtable.2 = {{.*}}, !type ![[TYPE2:[0-9]+]], !vcall_visibility ![[VCALL_VIS2:[0-9]+]]

#![crate_type = "lib"]

use std::rc::Rc;

trait T {
    // CHECK-LABEL: ; <virtual_function_elimination::S as virtual_function_elimination::T>::used
    fn used(&self) -> i32 {
        1
    }
    // CHECK-LABEL: ; <virtual_function_elimination::S as virtual_function_elimination::T>::used_through_sub_trait
    fn used_through_sub_trait(&self) -> i32 {
        3
    }
    // CHECK-LABEL: ; <virtual_function_elimination::S as virtual_function_elimination::T>::by_rc
    fn by_rc(self: Rc<Self>) -> i32 {
        self.used() + self.used()
    }
    // CHECK-LABEL-NOT: {{.*}}::unused
    fn unused(&self) -> i32 {
        2
    }
    // CHECK-LABEL-NOT: {{.*}}::by_rc_unused
    fn by_rc_unused(self: Rc<Self>) -> i32 {
        self.by_rc()
    }
}

trait U: T {
    // CHECK-LABEL: ; <virtual_function_elimination::S as virtual_function_elimination::U>::subtrait_used
    fn subtrait_used(&self) -> i32 {
        4
    }
    // CHECK-LABEL-NOT: {{.*}}::subtrait_unused
    fn subtrait_unused(&self) -> i32 {
        5
    }
}

pub trait V {
    // CHECK-LABEL: ; <virtual_function_elimination::S as virtual_function_elimination::V>::public_function
    fn public_function(&self) -> i32;
}

#[derive(Copy, Clone)]
struct S;

impl T for S {}

impl U for S {}

impl V for S {
    fn public_function(&self) -> i32 {
        6
    }
}

fn taking_t(t: &dyn T) -> i32 {
    // CHECK: @llvm.type.checked.load({{.*}}, i32 24, metadata !"[[MANGLED_TYPE0:[0-9a-zA-Z_]+]]")
    t.used()
}

fn taking_rc_t(t: Rc<dyn T>) -> i32 {
    // CHECK: @llvm.type.checked.load({{.*}}, i32 40, metadata !"[[MANGLED_TYPE0:[0-9a-zA-Z_]+]]")
    t.by_rc()
}

fn taking_u(u: &dyn U) -> i32 {
    // CHECK: @llvm.type.checked.load({{.*}}, i32 64, metadata !"[[MANGLED_TYPE1:[0-9a-zA-Z_]+]]")
    // CHECK: @llvm.type.checked.load({{.*}}, i32 24, metadata !"[[MANGLED_TYPE1:[0-9a-zA-Z_]+]]")
    // CHECK: @llvm.type.checked.load({{.*}}, i32 32, metadata !"[[MANGLED_TYPE1:[0-9a-zA-Z_]+]]")
    u.subtrait_used() + u.used() + u.used_through_sub_trait()
}

pub fn taking_v(v: &dyn V) -> i32 {
    // CHECK: @llvm.type.checked.load({{.*}}, i32 24, metadata !"NtC[[CRATE_IDENT:[a-zA-Z0-9]{12}]]_28virtual_function_elimination1V")
    v.public_function()
}

pub fn main() {
    let s = S;
    taking_t(&s);
    taking_rc_t(Rc::new(s));
    taking_u(&s);
    taking_v(&s);
}

// CHECK: ![[TYPE0]] = !{i64 0, !"[[MANGLED_TYPE0]]"}
// CHECK: ![[VCALL_VIS0]] = !{i64 2}
// CHECK: ![[TYPE1]] = !{i64 0, !"[[MANGLED_TYPE1]]"}
// CHECK: ![[TYPE2]] = !{i64 0, !"NtC[[CRATE_IDENT]]_28virtual_function_elimination1V"}
// CHECK: ![[VCALL_VIS2]] = !{i64 1}
