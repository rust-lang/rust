//@ aux-crate:priv:path_provenance_leaf=path_provenance_leaf.rs
//@ aux-crate:path_provenance_facade=path_provenance_facade.rs
//@ aux-crate:priv:path_provenance_private_facade=path_provenance_private_facade.rs
//@ compile-flags: -Zunstable-options

#![crate_type = "lib"]
#![deny(exported_private_dependencies)]
#![feature(trait_alias)]

extern crate path_provenance_facade as public_facade;
extern crate path_provenance_leaf as private_leaf;
extern crate path_provenance_private_facade as private_facade;

use private_facade::nested;
use private_facade::nested as renamed_nested;
use renamed_nested::T as Continued;

mod forwarding {
    pub use crate::private_facade::nested::U as Forwarded;
}

use forwarding::Forwarded;

pub fn direct() -> private_leaf::T {
    loop {}
}

pub fn public_reexport() -> public_facade::T {
    loop {}
}

pub fn private_reexport() -> private_facade::T {
    loop {}
}

pub fn imported_module() -> nested::T {
    loop {}
}

pub fn renamed_and_continued() -> Continued {
    loop {}
}

pub fn local_forwarding() -> Forwarded {
    loop {}
}

pub fn public_alias() -> public_facade::Alias {
    loop {}
}

pub fn private_alias() -> private_facade::Alias {
    loop {}
}

pub fn explicit_arg() -> public_facade::Generic<private_leaf::U> {
    loop {}
}

pub fn public_default() -> public_facade::Defaulted {
    loop {}
}

pub fn private_default() -> private_facade::Defaulted {
    loop {}
}

pub fn private_nominal_default() -> private_facade::Nominal {
    //~^ ERROR type `private_facade::Nominal` from private dependency
    loop {}
}

type LocalPrivate = private_leaf::T;
type LocalPublic = public_facade::Alias;
type LocalDefault<X = private_leaf::U> = X;

pub fn local_private_alias() -> LocalPrivate {
    loop {}
}

pub fn local_public_alias() -> LocalPublic {
    loop {}
}

pub fn local_omitted_default() -> LocalDefault {
    loop {}
}

pub fn local_explicit_default() -> LocalDefault<()> {
    loop {}
}

#[allow(exported_private_dependencies)]
pub struct LocalNominal<X = private_leaf::U>(X);

pub fn local_nominal_default() -> LocalNominal {
    loop {}
}

pub fn local_nominal_explicit() -> LocalNominal<()> {
    loop {}
}

pub fn direct_bound<X: private_leaf::Tr>() {}

pub fn public_bound<X: public_facade::Tr>() {}

pub fn private_bound<X: private_facade::Tr>() {}

pub fn associated_constraint<
    X: private_facade::Tr<Assoc = private_leaf::U>,
>() {
}

pub fn dyn_bound() -> Box<dyn private_facade::Tr<Assoc = ()>> {
    loop {}
}

pub fn impl_bound() -> impl private_facade::Tr {
    private_leaf::T
}

pub type Projection = <private_facade::T as private_facade::Tr>::Assoc;

pub fn public_trait_alias<X: public_facade::TraitAlias>() {}

pub fn private_trait_alias<X: private_facade::TraitAlias>() {}
//~^ ERROR trait `private_facade::TraitAlias` from private dependency

#[allow(exported_private_dependencies)]
pub trait LocalTraitAlias = private_leaf::Tr;

pub fn local_trait_alias<X: LocalTraitAlias>() {}

pub fn public_const_default() -> public_facade::ConstDefault {
    loop {}
}

pub fn private_const_default() -> private_facade::ConstDefault {
    loop {}
}

pub fn explicit_const_arg() -> public_facade::ConstDefault<{ private_leaf::N }> {
    loop {}
}

type LocalConstDefault<const N: usize = { private_leaf::N }> = public_facade::Const<N>;

pub fn local_const_default() -> LocalConstDefault {
    loop {}
}

pub fn local_const_explicit() -> LocalConstDefault<0> {
    loop {}
}

pub fn private_primitive_alias() -> private_facade::Primitive {
    loop {}
}
