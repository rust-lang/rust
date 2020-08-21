#![deny(broken_intra_doc_links)]
//~^ NOTE lint level is defined

// FIXME: this should say that it was skipped (maybe an allowed by default lint?)
/// [<invalid syntax>]

// FIXME: this could say which path was the first to not be found (in this case, `path`)
/// [path::to::nonexistent::module]
//~^ ERROR unresolved link
//~| NOTE no item named `path::to` is in scope
//~| HELP to escape

/// [f::A]
//~^ ERROR unresolved link
//~| NOTE this link partially resolves
//~| NOTE `f` is a function, not a module

/// [S::A]
//~^ ERROR unresolved link
//~| NOTE this link partially resolves
//~| NOTE no `A` in `S`

/// [S::fmt]
//~^ ERROR unresolved link
//~| NOTE this link partially resolves
//~| NOTE no `fmt` in `S`

/// [E::D]
//~^ ERROR unresolved link
//~| NOTE this link partially resolves
//~| NOTE no `D` in `E`

/// [u8::not_found]
//~^ ERROR unresolved link
//~| NOTE the builtin type `u8` does not have an associated item named `not_found`

/// [S!]
//~^ ERROR unresolved link
//~| HELP to link to the struct, use its disambiguator
//~| NOTE this link resolves to the struct `S`
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
//~^ ERROR unresolved link
//~| NOTE no item named `T::h`
//~| HELP to escape
pub trait T {
    fn g() {}
}
