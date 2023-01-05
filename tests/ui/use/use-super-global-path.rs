#![allow(unused)]

struct S;
struct Z;

mod foo {
    use ::super::{S, Z}; //~ ERROR global paths cannot start with `super`
                         //~| ERROR global paths cannot start with `super`

    pub fn g() {
        use ::super::main; //~ ERROR global paths cannot start with `super`
        main();
    }
}

fn main() { foo::g(); }
