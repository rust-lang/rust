//@ dont-require-annotations: ERROR
//@ compile-flags: --crate-type lib -Z ui-testing=no

#![feature(rustc_attrs)]

#[rustc_dump_object_lifetime_defaults]
struct Ref<'a, T: 'a>(&'a T);
