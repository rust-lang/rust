#![feature(rustc_attrs)]
#![crate_type = "lib"]

#[diagnostic::rustc_on_unimplemented(message = "Foo")]
pub struct Struct;
//~^ ERROR `#[diagnostic::rustc_on_unimplemented]` is only supported on trait definitions

#[diagnostic::rustc_on_unimplemented(on(Self = "IDontExist", message = "Foo"))]
//~^ ERROR cannot find type `IDontExist` in this scope [E0425]
//~| ERROR cannot find type `IDontExist`
pub trait Trait {}

mod module {
    pub struct Baz;
}

#[diagnostic::rustc_on_unimplemented(on(Self = "module::Baz", message = "Foo"))]
//~^ ERROR expected an identifier inside this `on`-clause
pub trait MultiSegmentNotSupported {}

pub struct Buz<A>{
    pub a: A
}

#[diagnostic::rustc_on_unimplemented(on(Self = "Buz<u8>", message = "Foo"))]
//~^ ERROR expected an identifier inside this `on`-clause
pub trait GenericNotSupported {}

#[diagnostic::rustc_on_unimplemented(on(Self = "u8", message = "Foo"))]
//~^ ERROR `u8` refers to a builtin type, not a struct, enum or union
pub trait PrimTyNotSupported {}


macro_rules! produce_trait {
    ($name: ident) =>{
        pub trait $name {}
    }
}

// FWIW this has never worked for any diagnostic attribute
#[diagnostic::rustc_on_unimplemented(message = "baz")]
produce_trait!(X);
//~^ERROR `#[diagnostic::rustc_on_unimplemented]` is only supported on trait definitions
