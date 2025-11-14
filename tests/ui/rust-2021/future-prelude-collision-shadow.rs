//@ check-pass
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
        // Here, `TryIntoU32` is imported and shadowed, but its methods are still available.
        let _: u32 = 3u8.try_into().unwrap();
        //~^ WARN trait method `try_into` will become ambiguous in Rust 2021
        //~| WARN this is accepted in the current edition
    }
}

fn main() {}
