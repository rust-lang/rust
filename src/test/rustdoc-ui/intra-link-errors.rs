#![deny(broken_intra_doc_links)]
//~^ NOTE lint level is defined

// FIXME: this should say that it was skipped (maybe an allowed by default lint?)
/// [invalid intra-doc syntax!!]

/// [path::to::nonexistent::module]
//~^ ERROR unresolved link
//~| NOTE no item named `path` in scope

/// [path::to::nonexistent::macro!]
//~^ ERROR unresolved link
//~| NOTE no item named `path` in scope

/// [type@path::to::nonexistent::type]
//~^ ERROR unresolved link
//~| NOTE no item named `path` in scope

/// [std::io::not::here]
//~^ ERROR unresolved link
//~| NOTE no item named `not` in module `io`

/// [type@std::io::not::here]
//~^ ERROR unresolved link
//~| NOTE no item named `not` in module `io`

/// [std::io::Error::x]
//~^ ERROR unresolved link
//~| NOTE the struct `Error` has no field

/// [std::io::ErrorKind::x]
//~^ ERROR unresolved link
//~| NOTE the enum `ErrorKind` has no variant

/// [f::A]
//~^ ERROR unresolved link
//~| NOTE `f` is a function, not a module

/// [f::A!]
//~^ ERROR unresolved link
//~| NOTE `f` is a function, not a module

/// [S::A]
//~^ ERROR unresolved link
//~| NOTE struct `S` has no field or associated item

/// [S::fmt]
//~^ ERROR unresolved link
//~| NOTE struct `S` has no field or associated item

/// [E::D]
//~^ ERROR unresolved link
//~| NOTE enum `E` has no variant or associated item

/// [u8::not_found]
//~^ ERROR unresolved link
//~| NOTE the builtin type `u8` has no associated item named `not_found`

/// [std::primitive::u8::not_found]
//~^ ERROR unresolved link
//~| NOTE the builtin type `u8` has no associated item named `not_found`

/// [type@Vec::into_iter]
//~^ ERROR unresolved link
//~| HELP to link to the associated function, add parentheses
//~| NOTE this link resolves to the associated function `into_iter`

/// [S!]
//~^ ERROR unresolved link
//~| HELP to link to the struct, prefix with `struct@`
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
//~| NOTE `T` has no macro named `h`
pub trait T {
    fn g() {}
}

/// [m()]
//~^ ERROR unresolved link
//~| HELP to link to the macro
//~| NOTE not in the value namespace
#[macro_export]
macro_rules! m {
    () => {};
}
