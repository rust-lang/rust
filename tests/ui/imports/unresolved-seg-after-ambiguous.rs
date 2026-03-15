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
//~^ ERROR: `E` is ambiguous

fn main() {}
