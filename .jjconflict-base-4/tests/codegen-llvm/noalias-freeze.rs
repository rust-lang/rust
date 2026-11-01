//@ compile-flags: -Copt-level=1

// References returned by a Frozen pointer type
// could be marked as "noalias", which caused miscompilation errors.
// This test runs the most minimal possible code that can reproduce this bug,
// and checks that noalias does not appear.
// See https://github.com/rust-lang/rust/issues/46239

#![crate_type = "lib"]

fn project<T>(x: &(T,)) -> &T {
    &x.0
}

fn dummy() {}

// CHECK-LABEL: @foo(
// CHECK-NOT: noalias
#[no_mangle]
pub fn foo() {
    let f = (dummy as fn(),);
    (*project(&f))();
}
