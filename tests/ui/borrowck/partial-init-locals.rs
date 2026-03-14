#![feature(partial_init_locals)]

struct A(u8);
struct B(A);
struct C(B);

impl Drop for C {
    fn drop(&mut self) {}
}

fn foo() -> u8 {
    2
}

fn main() {
    let a: A;
    a.0 = 1;
    let _ = &a.0;
    let _ = &a;
    let b: B;
    b.0.0 = foo();
    let _ = &b;

    let c: C;
    c.0.0.0 = 1; //~ ERROR: assigned binding `c` isn't fully initialized
}
