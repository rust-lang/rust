// Cloning small nested enums should not require branching code.

//@ compile-flags: -Copt-level=3

#![crate_type = "lib"]

#[derive(Clone)]
pub enum Foo {
    A(u8),
    B(bool),
}

#[derive(Clone)]
pub enum Bar {
    C(Foo),
    D(u8),
}

// CHECK-LABEL: @clone_foo
// CHECK-NOT: br i1
// CHECK-NOT: switch
// CHECK: ret
#[no_mangle]
pub fn clone_foo(f: &Foo) -> Foo {
    f.clone()
}

// CHECK-LABEL: @clone_bar
// CHECK-NOT: br i1
// CHECK-NOT: switch
// CHECK: ret
#[no_mangle]
pub fn clone_bar(b: &Bar) -> Bar {
    b.clone()
}
