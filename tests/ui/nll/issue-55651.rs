//@ check-pass

use std::mem::ManuallyDrop;

struct A;
struct B;

union U {
    a: ManuallyDrop<A>,
    b: ManuallyDrop<B>,
}

fn main() {
    unsafe {
        {
            let mut u = U { a: ManuallyDrop::new(A) };
            let a = u.a;
            u.a = ManuallyDrop::new(A);
            let a = u.a; // OK
        }
        {
            let mut u = U { a: ManuallyDrop::new(A) };
            let a = u.a;
            u.b = ManuallyDrop::new(B);
            let a = u.a; // OK
        }
    }
}
