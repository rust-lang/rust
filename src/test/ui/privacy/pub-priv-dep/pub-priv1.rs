 // aux-build:priv_dep.rs
#![feature(public_private_dependencies)]
#![deny(leaked_private_dependency)]

// This crate is a private dependency
extern crate priv_dep;

use priv_dep::OtherType;

// Type from private dependency used in private
// type - this is fine
struct PrivateType {
    field: OtherType
}

pub struct PublicType {
    pub field: OtherType,
    //~^ ERROR type `priv_dep::OtherType` from private dependency 'priv_dep' in public interface [leaked_private_dependency]
    //~| WARNING this was previously accepted
    priv_field: OtherType,
}

impl PublicType {
    pub fn pub_fn(param: OtherType) {}
    //~^ ERROR type `priv_dep::OtherType` from private dependency 'priv_dep' in public interface [leaked_private_dependency]
    //~| WARNING this was previously accepted

    fn priv_fn(param: OtherType) {}
}

fn main() {}
