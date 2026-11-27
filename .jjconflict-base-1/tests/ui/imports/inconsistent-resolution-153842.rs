mod a {
    pub use crate::s::Trait as s;
    //~^ ERROR cannot determine resolution for the import
    //~| ERROR cannot determine resolution for the import
    //~| ERROR unresolved imports `crate::s::Trait`, `a::s`
}

mod b {
    pub mod s {
        pub trait Trait {}
    }
}

use a::s;
use b::*;

fn main() {}
