// build-pass (FIXME(62277): could be check-pass?)

#![feature(untagged_unions)]

struct A;
struct B;

union U {
    a: A,
    b: B,
}

fn main() {
    unsafe {
        {
            let mut u = U { a: A };
            let a = u.a;
            u.a = A;
            let a = u.a; // OK
        }
        {
            let mut u = U { a: A };
            let a = u.a;
            u.b = B;
            let a = u.a; // OK
        }
    }
}
