//@ compile-flags: -Copt-level=0
//@ only-x86_64
//@ build-pass
//@ ignore-backends: gcc

#[repr(C, align(536870912))]
pub struct A(i64);

pub extern "C" fn foo(x: A) {}

fn main() {
    foo(A(0));
}
