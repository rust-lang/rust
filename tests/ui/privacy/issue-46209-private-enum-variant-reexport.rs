#[deny(unused_imports)]
mod rank {
    pub use self::Professor::*;
    //~^ ERROR glob import doesn't reexport anything
    //~| ERROR unused import: `self::Professor::*`
    pub use self::Lieutenant::{JuniorGrade, Full};
    //~^ ERROR `JuniorGrade` is private, and cannot be re-exported
    //~| ERROR `Full` is private, and cannot be re-exported
    //~| ERROR unused imports: `Full` and `JuniorGrade`
    pub use self::PettyOfficer::*;
    //~^ ERROR glob import doesn't reexport anything
    //~| ERROR unused import: `self::PettyOfficer::*`
    pub use self::Crewman::*;
    //~^ ERROR glob import doesn't reexport anything
    //~| ERROR unused import: `self::Crewman::*`

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

    pub(in crate::rank) enum PettyOfficer {
        SecondClass,
        FirstClass,
        Chief,
        MasterChief
    }

    pub(crate) enum Crewman {
        Recruit,
        Apprentice,
        Full
    }

}

fn main() {}
