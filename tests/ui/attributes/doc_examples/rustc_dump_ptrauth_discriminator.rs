//@ dont-require-annotations: ERROR
//@ compile-flags: --crate-type lib -Z ui-testing=no

#![feature(rustc_attrs)]

#[rustc_dump_ptrauth_discriminator(ptrauth_encoding, ptrauth_hash)]
extern "C" fn foo(_: i32) {}
