//@ aux-build:minicore.rs
//@ compile-flags: --crate-type=lib -Znext-solver -Cpanic=abort
//@ check-pass

#![feature(no_core, const_trait_impl)]
#![no_std]
#![no_core]

extern crate minicore;
use minicore::*;

fn is_const_fn<F>(_: F)
where
    F: const FnOnce(),
{
}

const fn foo() {}

fn test() {
    is_const_fn(foo);
}
