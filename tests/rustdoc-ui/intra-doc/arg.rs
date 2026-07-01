#![deny(rustdoc::broken_intra_doc_links)]
#![feature(intra_doc_arg)]

/// [arg@x]
//~^ ERROR argument `x` does not exist
pub fn a(y: ()) {}

pub struct X;

impl X {
    /// [arg@x]
    //~^ ERROR argument `x` does not exist
    pub fn a(y: ()) {}
}

pub trait T {
    /// [arg@x]
    //~^ ERROR argument `x` does not exist
    fn a(y: ()) {}

    /// [arg@x]
    //~^ ERROR argument `x` does not exist
    fn b(y: ());
}

extern "C" {
    /// [arg@x]
    //~^ ERROR argument `x` does not exist
    pub fn x(y: ());
}

/// [arg@a1]
//~^ ERROR can only be used in the documentation of functions
pub struct Y;
