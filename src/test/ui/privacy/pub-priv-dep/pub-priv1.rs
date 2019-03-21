 // aux-build:priv_dep.rs
 // aux-build:pub_dep.rs
 // extern-private:priv_dep
#![deny(exported_private_dependencies)]

// This crate is a private dependency
extern crate priv_dep;
// This crate is a public dependenct
extern crate pub_dep;

use priv_dep::{OtherType, OtherTrait};
use pub_dep::PubType;

// Type from private dependency used in private
// type - this is fine
struct PrivateType {
    field: OtherType
}

pub struct PublicType {
    pub field: OtherType,
    //~^ ERROR type `priv_dep::OtherType` from private dependency 'priv_dep' in public interface
    priv_field: OtherType, // Private field - this is fine
    pub other_field: PubType // Type from public dependency - this is fine
}

impl PublicType {
    pub fn pub_fn(param: OtherType) {}
    //~^ ERROR type `priv_dep::OtherType` from private dependency 'priv_dep' in public interface

    fn priv_fn(param: OtherType) {}
}

pub trait MyPubTrait {
    type Foo: OtherTrait;
}
//~^^^ ERROR trait `priv_dep::OtherTrait` from private dependency 'priv_dep' in public interface

pub struct AllowedPrivType {
    #[allow(exported_private_dependencies)]
    pub allowed: OtherType
}



fn main() {}
