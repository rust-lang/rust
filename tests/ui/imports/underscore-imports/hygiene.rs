// Make sure that underscore imports have the same hygiene considerations as other imports.

//@ check-pass

#![feature(decl_macro)]
#![allow(noop_method_call)]

mod x {
    pub use std::ops::Deref as _;
}

macro glob_import() {
    pub use crate::x::*;
}

macro underscore_import() {
    use std::ops::DerefMut as _;
}

mod y {
    crate::glob_import!();
    crate::underscore_import!();
}

macro create_module($y:ident) {
    mod $y {
        crate::glob_import!();
        crate::underscore_import!();
    }
}

create_module!(z);

fn main() {
    use crate::y::*;
    use crate::z::*;
    glob_import!();
    underscore_import!();
    (&()).deref();
    (&mut ()).deref_mut();
}
