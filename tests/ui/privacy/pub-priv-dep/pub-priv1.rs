//@ aux-crate:priv:priv_dep=priv_dep.rs
//@ aux-build:pub_dep.rs
//@ aux-crate:priv:pm=pm.rs
//@ compile-flags: -Zunstable-options

// Basic behavior check of exported_private_dependencies from either a public
// dependency or a private one.

#![deny(exported_private_dependencies)]

// This crate is a private dependency
pub extern crate priv_dep;
//~^ ERROR crate `priv_dep` from private dependency 'priv_dep' is re-exported
// This crate is a public dependency
pub extern crate pub_dep;
// This crate is a private dependency
extern crate pm;

use priv_dep::{OtherTrait, OtherType};
use pub_dep::PubType;

// Type from private dependency used in private
// type - this is fine
struct PrivateType {
    field: OtherType,
}

pub struct PublicType {
    pub field: OtherType,
    //~^ ERROR type `OtherType` from private dependency 'priv_dep' in public interface
    priv_field: OtherType,    // Private field - this is fine
    pub other_field: PubType, // Type from public dependency - this is fine
}

pub struct PublicTuple(
    pub OtherType,
    //~^ ERROR type `OtherType` from private dependency 'priv_dep' in public interface [exported_private_dependencies]
    OtherType,
    pub PubType,
);

pub enum PublicEnum {
    OtherType,
    ActualOtherType(OtherType, PubType),
    //~^ ERROR type `OtherType` from private dependency 'priv_dep' in public interface [exported_private_dependencies]
    ActualOtherTypeStruct {
        field: OtherType,
        //~^ ERROR type `OtherType` from private dependency 'priv_dep' in public interface [exported_private_dependencies]
        other_field: PubType,
    },
}

pub struct PublicGenericType<T, U>(pub T, U);
pub type ReexportedPublicGeneric = PublicGenericType<OtherType, ()>;
//~^ ERROR type `OtherType` from private dependency 'priv_dep' in public interface
pub type ReexportedPrivateGeneric = PublicGenericType<(), OtherType>;
//~^ ERROR type `OtherType` from private dependency 'priv_dep' in public interface

pub struct PublicGenericBoundedType<T: OtherTrait>(T);
//~^ ERROR trait `OtherTrait` from private dependency 'priv_dep' in public interface

impl PublicType {
    pub fn pub_fn_param(param: OtherType) {}
    //~^ ERROR type `OtherType` from private dependency 'priv_dep' in public interface

    pub fn pub_fn_return() -> OtherType { OtherType }
    //~^ ERROR type `OtherType` from private dependency 'priv_dep' in public interface

    fn priv_fn(param: OtherType) {}
}

pub trait MyPubTrait {
    type Foo: OtherTrait;
    //~^ ERROR trait `OtherTrait` from private dependency 'priv_dep' in public interface

    fn required_impl_trait() -> impl OtherTrait;

    fn provided_impl_trait() -> impl OtherTrait { OtherType }

    fn required_concrete() -> OtherType;
    //~^ ERROR type `OtherType` from private dependency 'priv_dep' in public interface

    fn provided_concrete() -> OtherType { OtherType }
    //~^ ERROR type `OtherType` from private dependency 'priv_dep' in public interface
}

pub trait WithSuperTrait: OtherTrait {}
//~^ ERROR trait `OtherTrait` from private dependency 'priv_dep' in public interface

pub trait PubLocalTraitWithAssoc {
    type X;
}

pub struct PrivateAssoc;
impl PubLocalTraitWithAssoc for PrivateAssoc {
    type X = OtherType;
//~^ ERROR type `OtherType` from private dependency 'priv_dep' in public interface
}

pub fn in_bounds<T: OtherTrait>(x: T) { unimplemented!() }
//~^ ERROR trait `OtherTrait` from private dependency 'priv_dep' in public interface

pub fn private_return_impl_trait() -> impl OtherTrait { OtherType }
//~^ ERROR trait `OtherTrait` from private dependency 'priv_dep' in public interface

pub fn private_return() -> OtherType { OtherType }
//~^ ERROR type `OtherType` from private dependency 'priv_dep' in public interface

pub fn private_in_generic() -> std::num::Saturating<OtherType> { unimplemented!() }
//~^ ERROR type `OtherType` from private dependency 'priv_dep' in public interface

pub static STATIC: OtherType = OtherType;
//~^ ERROR type `OtherType` from private dependency 'priv_dep' in public interface

pub const CONST: OtherType = OtherType;
//~^ ERROR type `OtherType` from private dependency 'priv_dep' in public interface

pub type Alias = OtherType;
//~^ ERROR type `OtherType` from private dependency 'priv_dep' in public interface

pub type AliasOfAlias = priv_dep::PubPub;
//~^ ERROR type `OtherType` from private dependency 'priv_dep' in public interface

pub struct PublicWithPrivateImpl;

impl OtherTrait for PublicWithPrivateImpl {}
//~^ ERROR trait `OtherTrait` from private dependency 'priv_dep' in public interface

pub trait PubTraitOnPrivate {}

impl PubTraitOnPrivate for OtherType {}
//~^ ERROR type `OtherType` from private dependency 'priv_dep' in public interface
//~| ERROR type `OtherType` from private dependency 'priv_dep' in public interface

pub struct PublicWithStdImpl;

impl From<OtherType> for PublicWithStdImpl {
//~^ ERROR type `OtherType` from private dependency 'priv_dep' in public interface
    fn from(val: OtherType) -> Self { Self }
    //~^ ERROR type `OtherType` from private dependency 'priv_dep' in public interface
}

impl From<PublicWithStdImpl> for OtherType {
    //~^ ERROR type `OtherType` from private dependency 'priv_dep' in public interface
    //~| ERROR type `OtherType` from private dependency 'priv_dep' in public interface
    fn from(val: PublicWithStdImpl) -> Self { Self }
    //~^ ERROR type `OtherType` from private dependency 'priv_dep' in public interface
    //~| ERROR type `OtherType` from private dependency 'priv_dep' in public interface
}

pub struct AllowedPrivType {
    #[allow(exported_private_dependencies)]
    pub allowed: OtherType,
}

pub use priv_dep::m;
//~^ ERROR macro `m` from private dependency 'priv_dep' is re-exported
pub use pm::fn_like;
//~^ ERROR macro `fn_like` from private dependency 'pm' is re-exported
pub use pm::PmDerive;
//~^ ERROR macro `PmDerive` from private dependency 'pm' is re-exported
pub use pm::pm_attr;
//~^ ERROR macro `pm_attr` from private dependency 'pm' is re-exported

pub use priv_dep::E::V1;
//~^ ERROR variant `V1` from private dependency 'priv_dep' is re-exported

pub use priv_dep::Unit;
//~^ ERROR type alias `Unit` from private dependency 'priv_dep' is re-exported
pub use priv_dep::PubPub;
//~^ ERROR type alias `PubPub` from private dependency 'priv_dep' is re-exported
pub use priv_dep::PubPriv;
//~^ ERROR type alias `PubPriv` from private dependency 'priv_dep' is re-exported
pub use priv_dep::OtherType as Renamed;
//~^ ERROR struct `Renamed` from private dependency 'priv_dep' is re-exported

fn main() {}
