#![warn(clippy::pub_use)]
#![allow(unused_imports)]
#![no_main]

pub mod outer {
    mod inner {
        pub struct Test {}
    }
    // should be linted
    pub use inner::Test;
    //~^ ERROR: using `pub use`
}

// should not be linted
use std::fmt;
