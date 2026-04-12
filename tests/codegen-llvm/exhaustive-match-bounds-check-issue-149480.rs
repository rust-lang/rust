//@ compile-flags: -O
// Regression test for #149480: bounds check should be eliminated when
// indexing an array with the result of an exhaustive match over nested enums.

#![crate_type = "lib"]

pub enum Foo {
    A(A),
    B(B),
}
pub enum A { A0, A1, A2 }
pub enum B { B0, B1 }

// CHECK-LABEL: @bar
#[no_mangle]
pub fn bar(foo: Foo, arr: &[u8; 5]) -> u8 {
    let offset: usize = match foo {
        Foo::A(A::A0) => 0,
        Foo::A(A::A1) => 1,
        Foo::A(A::A2) => 2,
        Foo::B(B::B0) => 3,
        Foo::B(B::B1) => 4,
    };
    // CHECK-NOT: panic_bounds_check
    arr[offset]
}
