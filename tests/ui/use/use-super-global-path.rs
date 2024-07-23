#![allow(unused)]

struct S;
struct Z;

mod foo {
    use ::super::{S, Z}; //~ ERROR cannot find module `super`
                         //~| ERROR cannot find module `super`

    pub fn g() {
        use ::super::main; //~ ERROR cannot find module `super`
        main();
    }
}

fn main() { foo::g(); }
