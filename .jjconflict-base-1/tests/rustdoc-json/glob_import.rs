// This is a regression test for <https://github.com/rust-lang/rust/issues/98003>.

#![no_std]

//@ has "$.index[?(@.name=='glob')]"
//@ has "$.index[?(@.inner.use)].inner.use.name" \"*\"

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
