//@ proc-macro: test-macros.rs
#![crate_type = "lib"]
#![deny(unused_imports)]

extern crate test_macros;

mod other {
    pub use test_macros::{Empty, empty, empty_attr};

    pub trait Empty {}
}

pub use other::{Empty, empty, empty_attr};
pub use test_macros::{Empty, empty, empty_attr}; //~ ERROR unused imports
