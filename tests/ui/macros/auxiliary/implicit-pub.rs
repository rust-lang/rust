//@ edition:2018
#[allow(unused_macros)]
macro_rules! fake_pub {
    () => {}
}

#[macro_export]
macro_rules! real_pub {
    () => {}
}

pub mod inner {
    pub use real_pub;
}

pub use real_pub as real_pub_reexport;
