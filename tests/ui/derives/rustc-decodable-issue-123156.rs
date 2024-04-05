//@ check-pass
//@ edition:2021
//@ aux-build:rustc-serialize.rs

#![crate_type = "lib"]
#![allow(deprecated, soft_unstable)]

extern crate rustc_serialize;

#[derive(RustcDecodable)]
pub enum Foo {}
