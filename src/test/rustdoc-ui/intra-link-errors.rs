#![deny(broken_intra_doc_links)]
//~^ NOTE lint level is defined

//! [std::io::oops]
//! [std::io::oops::not::here]

// FIXME: this should say that it was skipped (maybe an allowed by default lint?)
/// [<invalid syntax>]

// FIXME: this could say which path was the first to not be found (in this case, `path`)
/// [path::to::nonexistent::module]
//~^ ERROR unresolved link
//~| NOTE no item named `path::to::nonexistent` is in scope
//~| HELP to escape

// TODO: why does this say `f` and not `f::A`??
/// [f::A]
//~^ ERROR unresolved link
//~| NOTE no item named `f` is in scope
//~| HELP to escape

/// [S::A]
//~^ ERROR unresolved link
//~| NOTE this link partially resolves
//~| NOTE `S` has no field

/// [S::fmt]
//~^ ERROR unresolved link
//~| NOTE this link partially resolves
//~| NOTE `S` has no field

/// [E::D]
//~^ ERROR unresolved link
//~| NOTE this link partially resolves
//~| NOTE `E` has no field

/// [u8::not_found]
//~^ ERROR unresolved link
//~| NOTE the builtin type `u8` does not have an associated item named `not_found`

/// [S!]
//~^ ERROR unresolved link
//~| HELP to link to the unit struct, use its disambiguator
//~| NOTE this link resolves to the unit struct `S`
pub fn f() {}
#[derive(Debug)]
pub struct S;

pub enum E { A, B, C }

/// [type@S::h]
//~^ ERROR unresolved link
//~| HELP to link to the associated function
//~| NOTE not in the type namespace
impl S {
    pub fn h() {}
}

/// [type@T::g]
//~^ ERROR unresolved link
//~| HELP to link to the associated function
//~| NOTE not in the type namespace

/// [T::h!]
pub trait T {
    fn g() {}
}
