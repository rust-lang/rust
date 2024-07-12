#![allow(unused)]

struct S;
struct Z;

mod foo {
    use ::super::{S, Z}; //~ ERROR cannot find module `super`
                         //~| ERROR cannot find module `super`
                         //~| NOTE global paths cannot start with `super`
                         //~| NOTE global paths cannot start with `super`
                         //~| NOTE duplicate diagnostic

    pub fn g() {
        use ::super::main; //~ ERROR cannot find module `super`
                           //~| NOTE global paths cannot start with `super`
        main();
    }
}

fn main() { foo::g(); }
