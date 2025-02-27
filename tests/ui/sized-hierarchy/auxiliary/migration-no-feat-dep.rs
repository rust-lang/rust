//@ compile-flags: -Zunstable-options
//@ edition: future
#![feature(const_trait_impl, sized_hierarchy)]

pub fn needs_const_sized<T: const Sized>() { unimplemented!() }
pub fn needs_const_metasized<T: const MetaSized>() { unimplemented!() }
