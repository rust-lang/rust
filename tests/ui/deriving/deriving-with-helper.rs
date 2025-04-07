//@ add-core-stubs
//@ check-pass
//@ compile-flags: --crate-type=lib

#![feature(decl_macro)]
#![feature(lang_items)]
#![feature(no_core)]
#![feature(rustc_attrs)]

#![no_core]

extern crate minicore;
use minicore::*;

#[rustc_builtin_macro]
macro derive() {}

#[rustc_builtin_macro(Default, attributes(default))]
macro Default() {}

mod default {
    pub trait Default {
        fn default() -> Self;
    }

    impl Default for u8 {
        fn default() -> u8 {
            0
        }
    }
}

#[derive(Default)]
enum S {
    #[default] // OK
    Foo,
}
