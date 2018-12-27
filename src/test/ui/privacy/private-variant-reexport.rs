mod m1 {
    pub use ::E::V; //~ ERROR variant `V` is private and cannot be re-exported
}

mod m2 {
    pub use ::E::{V}; //~ ERROR variant `V` is private and cannot be re-exported
}

mod m3 {
    pub use ::E::V::{self}; //~ ERROR variant `V` is private and cannot be re-exported
}

mod m4 {
    pub use ::E::*; //~ ERROR enum is private and its variants cannot be re-exported
}

enum E { V }

fn main() {}
