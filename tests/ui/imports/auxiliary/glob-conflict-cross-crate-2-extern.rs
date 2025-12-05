mod a {
    pub type C = i8;
}

mod b {
    pub type C = i16;
}

pub use a::*;
pub use b::*;
