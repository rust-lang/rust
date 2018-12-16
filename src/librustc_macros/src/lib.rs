#![feature(proc_macro_hygiene)]

#[macro_use]
extern crate syn;
#[macro_use]
extern crate synstructure;
#[macro_use]
extern crate quote;
extern crate proc_macro;
extern crate proc_macro2;

mod hash_stable;

decl_derive!([HashStable, attributes(stable_hasher)] => hash_stable::hash_stable_derive);
