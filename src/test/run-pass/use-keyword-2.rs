#![allow(unused_variables)]
pub struct A;

mod test {
    pub use super :: A;

    pub use self :: A as B;
}

impl A {
    fn f() {}
    fn g() {
        Self :: f()
    }
}

fn main() {
    let a: A = test::A;
    let b: A = test::B;
    let c: () = A::g();
}
