// run-pass
#![allow(dead_code)]
#![allow(unused_imports)]
#![allow(non_snake_case)]

mod A {
    pub mod B {
        use super::*;

        pub struct S;
    }

    pub mod C {
        use super::*;
        use super::B::S;

        pub struct T;
    }

    pub use self::C::T;
}

use A::*;

fn main() {}
