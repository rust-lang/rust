#![feature(prelude_import)]
#![no_std]
extern crate std;
#[prelude_import]
use ::std::prelude::rust_2015::*;
//@ pretty-mode:expanded
//@ pp-exact:block-index-paren.pp

macro_rules! block_arr { () => {{ [0u8; 4] }}; }

macro_rules! as_slice { () => {{ &block_arr!()[..] }}; }

macro_rules! group { ($e:expr) => { $e }; }

fn scope() { &({ drop })(0); }

fn main() { let _: &[u8] = { &({ [0u8; 4] })[..] }; }
