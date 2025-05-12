#![allow(ambiguous_glob_reexports)]

mod m1 {
    pub fn f() {}
}
mod m2 {
    pub fn f(_: u8) {}
}

pub use m1::*;
pub use m2::*;

pub mod glob {
    pub use *;
}
