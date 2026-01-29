//@ edition:2018
#![feature(macro_implicit_pub)]
#[allow(unused_macros)]
macro_rules! fake_pub {
    () => {}
}

macro_rules! real_pub {
    () => {}
}

pub mod inner {
    pub use real_pub;
}

pub use real_pub as real_pub_reexport;
