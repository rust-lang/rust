//@ edition:2015
mod m {
    pub enum E {
        Fn(u8),
        Struct {
            s: u8,
        },
        Unit,
    }

    pub mod n {
        pub(in crate::m) enum Z {
            Fn(u8),
            Struct {
                s: u8,
            },
            Unit,
        }
    }

    use crate::m::n::Z; // OK, only the type is imported

    fn f() {
        n::Z;
        //~^ ERROR cannot find value `Z` in module `n`
        Z;
        //~^ ERROR cannot find value `Z` in this scope
        let _: Z = Z::Fn;
        //~^ ERROR mismatched types
        let _: Z = Z::Struct;
        //~^ ERROR expected value, found struct variant `Z::Struct`
        let _ = Z::Unit();
        //~^ ERROR expected function, found enum variant `Z::Unit`
        let _ = Z::Unit {};
        // This is ok, it is equivalent to not having braces
    }
}

use m::E; // OK, only the type is imported

fn main() {
    let _: E = m::E;
    //~^ ERROR cannot find value `E` in module `m`
    let _: E = m::E::Fn;
    //~^ ERROR mismatched types
    let _: E = m::E::Struct;
    //~^ ERROR expected value, found struct variant `m::E::Struct`
    let _: E = m::E::Unit();
    //~^ ERROR expected function, found enum variant `m::E::Unit`
    let _: E = E;
    //~^ ERROR cannot find value `E` in this scope
    let _: E = E::Fn;
    //~^ ERROR mismatched types
    let _: E = E::Struct;
    //~^ ERROR expected value, found struct variant `E::Struct`
    let _: E = E::Unit();
    //~^ ERROR expected function, found enum variant `E::Unit`
    let _: Z = m::n::Z;
    //~^ ERROR cannot find type `Z` in this scope
    //~| ERROR cannot find value `Z` in module `m::n`
    //~| ERROR enum `Z` is private
    let _: Z = m::n::Z::Fn;
    //~^ ERROR cannot find type `Z` in this scope
    //~| ERROR enum `Z` is private
    let _: Z = m::n::Z::Struct;
    //~^ ERROR cannot find type `Z` in this scope
    //~| ERROR expected value, found struct variant `m::n::Z::Struct`
    //~| ERROR enum `Z` is private
    let _: Z = m::n::Z::Unit {};
    //~^ ERROR cannot find type `Z` in this scope
    //~| ERROR enum `Z` is private
}
