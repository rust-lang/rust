//@ known-bug: #121444
//@ compile-flags: -Copt-level=0
//@ edition:2021
//@ only-x86_64
//@ ignore-windows
#[repr(align(536870912))]
pub struct A(i64);

pub extern "C" fn foo(x: A) {}

fn main() {
    foo(A(0));
}
