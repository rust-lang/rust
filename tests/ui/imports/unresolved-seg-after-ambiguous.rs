mod a {
    mod b {
        mod c {
            pub struct E;
        }

        mod d {
            #[derive(Debug)]
            pub struct E;
        }

        pub use self::d::*;
        pub use self::c::*;
    }

    pub use self::b::*;
}

use self::a::E::in_exist;
//~^ ERROR: unresolved import `self::a::E`
//~| ERROR: `E` is ambiguous
//~| WARNING: this was previously accepted by the compiler but is being phased out; it will become a hard error in a future release!

fn main() {}
