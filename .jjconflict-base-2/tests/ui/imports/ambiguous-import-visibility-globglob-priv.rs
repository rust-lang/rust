mod m {
    pub struct S {}
}

mod both {
    pub mod private {
        use crate::m::*;
        pub(super) use crate::m::*;
    }
}

use crate::both::private::S;
//~^ ERROR struct import `S` is private

// Separate module to make visibilities `in crate::inner` and `in crate::both(::private)` unordered.
mod inner {
    use crate::both::private::S;
    //~^ ERROR struct import `S` is private
}

fn main() {}
