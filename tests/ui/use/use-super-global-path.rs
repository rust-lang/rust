#![allow(unused)]

struct S;
struct Z;

mod foo {
    use ::super::{S, Z};
    //~^ ERROR: unresolved import `super`

    pub fn g() {
        use ::super::main; //~ ERROR: global paths cannot start with `super`
        main();
    }
}

fn main() { foo::g(); }
