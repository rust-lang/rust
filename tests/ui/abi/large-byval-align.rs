//@ compile-flags: -Copt-level=0
//@ only-x86_64
//@ build-pass

#[repr(align(536870912))]
pub struct A(i64);

#[allow(improper_ctypes_definitions)]
pub extern "C" fn foo(x: A) {}

fn main() {
    foo(A(0));
}
