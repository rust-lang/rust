// check-pass

#![deny(unreachable_pub)]

pub use self::m1::*;

mod m1 {
    pub use self::m2::*;

    mod m2 {
        pub struct Item1;
        pub struct Item2;
    }
}


pub use self::o1::{ Item42, Item24 };

mod o1 {
    pub use self::o2::{ Item42, Item24 };

    mod o2 {
        pub struct Item42;
        pub struct Item24;
    }
}

fn main() {}
