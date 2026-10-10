//@ edition:2015
//@ aux-build:privacy-struct-ctor.rs

extern crate privacy_struct_ctor as xcrate;

mod m {
    pub struct S(u8);
    pub struct S2 {
        s: u8
    }

    pub mod n {
        pub(in crate::m) struct Z(pub(in crate::m::n) u8);
    }

    use crate::m::n::Z; // OK, only the type is imported

    fn f() {
        n::Z;
        //~^ ERROR tuple struct constructor `Z` is private
        Z;
        //~^ ERROR cannot find value `Z` in this scope
    }
}

use m::S; // OK, only the type is imported
use m::S2; // OK, only the type is imported

fn main() {
    m::S;
    //~^ ERROR tuple struct constructor `S` is private
    let _: S = m::S(2);
    //~^ ERROR tuple struct constructor `S` is private
    S;
    //~^ ERROR cannot find value `S` in this scope
    m::n::Z;
    //~^ ERROR tuple struct constructor `Z` is private

    S2;
    //~^ ERROR cannot find value `S2` in this scope

    xcrate::m::S;
    //~^ ERROR tuple struct constructor `S` is private
    xcrate::S;
    //~^ ERROR cannot find value `S` in crate `xcrate`
    xcrate::m::n::Z;
    //~^ ERROR tuple struct constructor `Z` is private
}
