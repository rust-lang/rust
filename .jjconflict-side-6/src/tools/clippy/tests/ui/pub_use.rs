#![warn(clippy::pub_use)]
#![no_main]

pub mod outer {
    mod inner {
        pub struct Test {}
    }
    // should be linted
    pub use inner::Test;
    //~^ pub_use
}

// should not be linted
use std::fmt;
