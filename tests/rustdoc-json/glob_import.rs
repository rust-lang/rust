// This is a regression test for <https://github.com/rust-lang/rust/issues/98003>.

#![feature(no_core)]
#![no_std]
#![no_core]

// @has "$.index[*][?(@.name=='glob')]"
// @has "$.index[*][?(@.kind=='import')].inner.name" \"*\"


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
