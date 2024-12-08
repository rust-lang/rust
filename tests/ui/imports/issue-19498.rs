use self::A;
use self::B;
mod A {} //~ ERROR the name `A` is defined multiple times
//~| `A` redefined here
pub mod B {} //~ ERROR the name `B` is defined multiple times
//~| `B` redefined here
mod C {
    use C::D;
    mod D {} //~ ERROR the name `D` is defined multiple times
    //~| `D` redefined here
}

fn main() {}
