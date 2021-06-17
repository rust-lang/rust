// run-rustfix
// edition:2018
// check-pass
#![warn(future_prelude_collision)]
#![allow(dead_code)]

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
        //~^ ERROR no method name `try_into` found
    }
}

fn main() {}
