#![deny(unused_imports)]

mod a {}

pub use a::*;
//~^ ERROR: unused import: `a::*`

mod b {
    mod c {
        #[derive(Clone)]
        pub struct D;
    }
    pub use self::c::*; // don't show unused import lint
}

pub use b::*; // don't show unused import lint

mod d {
    const D: i32 = 1;
}

pub use d::*;
//~^ ERROR: unused import: `d::*`

fn main() {}
