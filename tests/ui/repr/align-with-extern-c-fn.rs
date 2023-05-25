// run-pass

// #45662

#[repr(C, align(16))]
pub struct A(i64);

pub extern "C" fn foo(_x: A) {}

fn main() {
    foo(A(0));
}
