//! Checks module re-exports, aliasing with `pub use`,
//! and calling private methods via `Self` in an impl block.

//@ run-pass

#![allow(unused_variables)]
pub struct A;

mod test {
    pub use self::A as B;
    pub use super::A;
}

impl A {
    fn f() {}
    fn g() {
        Self::f()
    }
}

fn main() {
    let a: A = test::A;
    let b: A = test::B;
    let c: () = A::g();
}
