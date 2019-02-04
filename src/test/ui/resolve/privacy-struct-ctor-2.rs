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
        //~^ ERROR tuple struct `n::Z` is private
    }
}

use m::S; // OK, only the type is imported
use m::S2; // OK, only the type is imported

fn main() {
    m::S;
    //~^ ERROR tuple struct `m::S` is private
    let _: S = m::S(2);
    //~^ ERROR field `0` of struct `m::S` is private
    m::n::Z;
    //~^ ERROR struct `m::n::Z` is private

    xcrate::m::S;
    //~^ ERROR struct `xcrate::S` is private
    xcrate::m::n::Z;
    //~^ ERROR struct `xcrate::m::n::Z` is private
}
