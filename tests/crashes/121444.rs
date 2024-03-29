//@ known-bug: #121444
//@ compile-flags: -Copt-level=0
//@ edition:2021
#[repr(align(536870912))]
pub struct A(i64);

pub extern "C" fn foo(x: A) {}

fn main() {
    foo(A(0));
}
