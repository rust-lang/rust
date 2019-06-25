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
