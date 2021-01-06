mod m1 {
    pub use ::E::V; //~ ERROR `V` is private, and cannot be re-exported
}

mod m2 {
    pub use ::E::{V}; //~ ERROR `V` is private, and cannot be re-exported
}

mod m3 {
    pub use ::E::V::{self}; //~ ERROR `V` is private, and cannot be re-exported
}

#[deny(unused_imports)]
mod m4 {
    pub use ::E::*; //~ ERROR glob import doesn't reexport anything
}

enum E { V }

fn main() {}
