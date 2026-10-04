//@ compile-flags: -Znext-solver

// Unsafe binders are `Send`/`Sync` if their inner type is.

#![feature(unsafe_binders)]
#![allow(incomplete_features, dead_code)]

use std::cell::Cell;

fn is_send<T: Send>() {}
fn is_sync<T: Sync>() {}

fn yes() {
    is_send::<unsafe<'a> &'a u8>();
    is_sync::<unsafe<'a> &'a u8>();
}

fn not_send() {
    is_send::<unsafe<'a> &'a Cell<u8>>();
    //~^ ERROR `Cell<u8>` cannot be shared between threads safely
}

fn not_sync() {
    is_sync::<unsafe<> *const u8>();
    //~^ ERROR `*const u8` cannot be shared between threads safely
}

fn main() {}
