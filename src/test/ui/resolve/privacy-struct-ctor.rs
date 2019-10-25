// aux-build:privacy-struct-ctor.rs

extern crate privacy_struct_ctor as xcrate;

mod m {
    pub struct S(u8);
    pub struct S2 {
        s: u8
    }

    pub mod n {
        pub(in m) struct Z(pub(in m::n) u8);
    }

    use m::n::Z; // OK, only the type is imported

    fn f() {
        n::Z;
        //~^ ERROR tuple struct `Z` is private
        Z;
        //~^ ERROR expected value, found struct `Z`
    }
}

use m::S; // OK, only the type is imported
use m::S2; // OK, only the type is imported

fn main() {
    m::S;
    //~^ ERROR tuple struct `S` is private
    let _: S = m::S(2);
    //~^ ERROR tuple struct `S` is private
    S;
    //~^ ERROR expected value, found struct `S`
    m::n::Z;
    //~^ ERROR tuple struct `Z` is private

    S2;
    //~^ ERROR expected value, found struct `S2`

    xcrate::m::S;
    //~^ ERROR tuple struct `S` is private
    xcrate::S;
    //~^ ERROR expected value, found struct `xcrate::S`
    xcrate::m::n::Z;
    //~^ ERROR tuple struct `Z` is private
}
