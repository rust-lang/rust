// Check that cyclic glob imports are allowed with underscore imports

//@ check-pass

#![allow(noop_method_call)]

mod x {
    pub use crate::y::*;
    pub use std::ops::Deref as _;
}

mod y {
    pub use crate::x::*;
    pub use std::ops::Deref as _;
}

pub fn main() {
    use x::*;
    (&0).deref();
}
