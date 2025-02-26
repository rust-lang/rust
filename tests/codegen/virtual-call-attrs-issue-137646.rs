//! Regression test for https://github.com/rust-lang/rust/issues/137646.
//! Since we don't know the exact implementation of the virtual call,
//! it might write to parameters, we can't infer the readonly attribute.
//@ compile-flags: -C opt-level=3 -C no-prepopulate-passes

#![crate_type = "lib"]

pub trait Trait {
    fn m(&self, _: (i32, i32, i32)) {}
}

#[no_mangle]
pub fn foo(trait_: &dyn Trait) {
    // CHECK-LABEL: @foo(
    // CHECK: call void
    // CHECK-NOT: readonly
    trait_.m((1, 1, 1));
}
