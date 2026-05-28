// Make sure that underscore imports with different contexts can exist in the
// same scope.

//@ check-pass

#![feature(decl_macro)]

mod x {
    pub use std::ops::Deref as _;
}

macro n() {
    pub use crate::x::*;
}

#[macro_export]
macro_rules! p {
    () => { pub use crate::x::*; }
}

macro m($y:ident) {
    mod $y {
        crate::n!(); // Reexport of `Deref` should not be imported in `main`
        crate::p!(); // Reexport of `Deref` should be imported into `main`
    }
}

m!(y);

fn main() {
    use crate::y::*;
    #[allow(noop_method_call)]
    (&()).deref();
}
