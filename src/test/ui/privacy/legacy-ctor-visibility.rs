// ignore-tidy-linelength

#![allow(unused)]

use m::S;

mod m {
    pub struct S(u8);

    mod n {
        use S;
        fn f() {
            S(10);
            //~^ ERROR private struct constructors are not usable through re-exports in outer modules
            //~| WARN this was previously accepted
        }
    }
}

fn main() {}
