//@ dont-require-annotations: NOTE

use self::A;
use self::B;
mod A {} //~ ERROR the name `A` is defined multiple times
//~| NOTE `A` redefined here
pub mod B {} //~ ERROR the name `B` is defined multiple times
//~| NOTE `B` redefined here
mod C {
    use crate::C::D;
    mod D {} //~ ERROR the name `D` is defined multiple times
    //~| NOTE `D` redefined here
}

fn main() {}
