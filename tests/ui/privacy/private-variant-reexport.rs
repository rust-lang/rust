mod m1 {
    pub use ::E::V; //~ ERROR `V` is only public within the crate, and cannot be re-exported outside
}

mod m2 {
    pub use ::E::{V}; //~ ERROR `V` is only public within the crate, and cannot be re-exported outside
}

mod m3 {
    pub use ::E::V::{self}; //~ ERROR `V` is only public within the crate, and cannot be re-exported outside
}

#[deny(unused_imports)]
mod m4 {
    pub use ::E::*;
    //~^ ERROR glob import doesn't reexport anything
    //~| ERROR unused import: `::E::*`
}

enum E { V }

fn main() {}
