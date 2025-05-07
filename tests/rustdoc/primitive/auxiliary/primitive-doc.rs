//@ compile-flags: --crate-type lib
//@ edition: 2018

#![feature(rustc_attrs)]
#![feature(no_core)]
#![no_core]

#[rustc_doc_primitive = "usize"]
/// This is the built-in type `usize`.
mod usize {
}
