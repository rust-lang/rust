//@ check-fail
//@ compile-flags: --crate-type=lib
//@ edition: 2024
//@ run-rustfix

#![deny(sized_hierarchy_migration)]
#![feature(const_trait_impl, sized_hierarchy)]

use std::marker::MetaSized;

// Test that use of `?Sized`, `Sized` or default supertrait/bounds trigger edition migration lints.

pub fn needs_const_sized<T: const Sized>() { unimplemented!() }
pub fn needs_const_metasized<T: const MetaSized>() { unimplemented!() }


pub fn relaxed_bound_migration<T: ?Sized>() {
//~^ ERROR `?Sized` bound relaxations are being migrated to `const MetaSized`
//~| WARN this is accepted in the current edition (Rust 2024) but is a hard error in Rust future
    needs_const_metasized::<T>()
}


pub fn default_sized_to_const_sized<T>() {
//~^ ERROR default bounds are being migrated to `const Sized`
//~| WARN this is accepted in the current edition (Rust 2024) but is a hard error in Rust future
    needs_const_sized::<T>()
}

pub fn sized_to_const_sized<T: Sized>() {
//~^ ERROR `Sized` bounds are being migrated to `const Sized`
//~| WARN this is accepted in the current edition (Rust 2024) but is a hard error in Rust future
    needs_const_sized::<T>()
}


pub trait ImplicitSupertrait {
//~^ ERROR a `const MetaSized` supertrait is required to maintain backwards compatibility
//~| WARN this is accepted in the current edition (Rust 2024) but is a hard error in Rust future
    fn check() {
        needs_const_metasized::<Self>()
    }
}

pub trait SizedToConstSized: Sized {
//~^ ERROR `Sized` bounds are being migrated to `const Sized`
//~| WARN this is accepted in the current edition (Rust 2024) but is a hard error in Rust future
    fn check() {
        needs_const_sized::<Self>()
    }
}

pub trait AssocType {
//~^ ERROR a `const MetaSized` supertrait is required to maintain backwards compatibility
//~| WARN this is accepted in the current edition (Rust 2024) but is a hard error in Rust future
    type Foo: Sized;
//~^ ERROR `Sized` bounds are being migrated to `const Sized`
//~| WARN this is accepted in the current edition (Rust 2024) but is a hard error in Rust future
//~^^^ ERROR `Sized` bounds are being migrated to `const Sized`
//~| WARN this is accepted in the current edition (Rust 2024) but is a hard error in Rust future
    type Bar: ?Sized;
//~^ ERROR `?Sized` bound relaxations are being migrated to `const MetaSized`
//~| WARN this is accepted in the current edition (Rust 2024) but is a hard error in Rust future
//~^^^ ERROR `?Sized` bound relaxations are being migrated to `const MetaSized`
//~| WARN this is accepted in the current edition (Rust 2024) but is a hard error in Rust future

    fn check() {
        needs_const_sized::<Self::Foo>();
        needs_const_metasized::<Self::Bar>();
    }
}
