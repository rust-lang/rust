//@ aux-build:migration-no-feat-dep.rs
//@ check-pass
//@ compile-flags: --crate-type=lib
//@ edition: 2024

#![feature(const_trait_impl)]

extern crate migration_no_feat_dep;
use migration_no_feat_dep::{needs_const_metasized, needs_const_sized};

// Test that use of `?Sized`, `Sized` or default supertrait/bounds without the `sized_hierarchy`
// feature work as expected with a migrated crate.

pub fn relaxed_bound_migration<T: ?Sized>() {
    needs_const_metasized::<T>()
}


pub fn default_sized_to_const_sized<T>() {
    needs_const_sized::<T>()
}

pub fn sized_to_const_sized<T: Sized>() {
    needs_const_sized::<T>()
}


pub trait ImplicitSupertrait {
    fn check() {
        needs_const_metasized::<Self>()
    }
}

pub trait SizedToConstSized: Sized {
    fn check() {
        needs_const_sized::<Self>()
    }
}

pub trait AssocType {
    type Foo: Sized;
    type Bar: ?Sized;

    fn check() {
        needs_const_sized::<Self::Foo>();
        needs_const_metasized::<Self::Bar>();
    }
}
