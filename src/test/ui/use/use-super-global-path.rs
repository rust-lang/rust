#![allow(unused)]

struct S;
struct Z;

mod foo {
    use ::super::{S, Z}; //~ ERROR global paths cannot start with `super`

    pub fn g() {
        use ::super::main; //~ ERROR global paths cannot start with `super`
        main(); //~ ERROR cannot find function `main` in this scope
    }
}

fn main() { foo::g(); }
