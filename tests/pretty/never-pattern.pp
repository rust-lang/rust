#![feature(prelude_import)]
#![no_std]
//@ pretty-mode:expanded
//@ pp-exact:never-pattern.pp
//@ only-x86_64

#![allow(incomplete_features)]
#![feature(never_patterns)]
#![feature(never_type)]
#[macro_use]
extern crate std;
#[prelude_import]
use ::std::prelude::rust_2015::*;

fn f(x: Result<u32, !>) { _ = match x { Ok(x) => x, Err(!) , }; }

fn main() {}
