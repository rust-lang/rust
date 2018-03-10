// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(crate_visibility_modifier)]

mod rank {
    pub use self::Professor::*;
    //~^ ERROR enum is private and its variants cannot be re-exported
    pub use self::Lieutenant::{JuniorGrade, Full};
    //~^ ERROR variant `JuniorGrade` is private and cannot be re-exported
    //~| ERROR variant `Full` is private and cannot be re-exported
    pub use self::PettyOfficer::*;
    //~^ ERROR enum is private and its variants cannot be re-exported
    pub use self::Crewman::*;
    //~^ ERROR enum is private and its variants cannot be re-exported

    enum Professor {
        Adjunct,
        Assistant,
        Associate,
        Full
    }

    enum Lieutenant {
        JuniorGrade,
        Full,
    }

    pub(in rank) enum PettyOfficer {
        SecondClass,
        FirstClass,
        Chief,
        MasterChief
    }

    crate enum Crewman {
        Recruit,
        Apprentice,
        Full
    }

}

fn main() {}
