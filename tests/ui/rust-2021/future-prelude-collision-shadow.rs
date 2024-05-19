//@ edition:2018
#![warn(rust_2021_prelude_collisions)]
#![allow(dead_code)]
#![allow(unused_imports)]

mod m {
    pub trait TryIntoU32 {
        fn try_into(self) -> Result<u32, ()>;
    }

    impl TryIntoU32 for u8 {
        fn try_into(self) -> Result<u32, ()> {
            Ok(self as u32)
        }
    }

    pub trait AnotherTrick {}
}

mod d {
    use crate::m::AnotherTrick as TryIntoU32;
    use crate::m::*;

    fn main() {
        // Here, `TryIntoU32` is imported but shadowed, but in that case we don't permit its methods
        // to be available.
        let _: u32 = 3u8.try_into().unwrap();
        //~^ ERROR no method named `try_into` found for type `u8` in the current scope
    }
}

fn main() {}
