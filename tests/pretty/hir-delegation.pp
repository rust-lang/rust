//@ pretty-compare-only
//@ pretty-mode:hir
//@ pp-exact:hir-delegation.pp

#![allow(incomplete_features)]#![feature(fn_delegation)]
#[prelude_import]
use ::std::prelude::rust_2015::*;
#[macro_use]
extern crate std;

fn b<C>(e: C) { }

trait G {
    fn b(arg0: _) -> _ { b({ }) }
}

mod m {
    fn add(a: u32, b: u32) -> u32 { a + b }
}

fn add(arg0: _, arg1: _) -> _ { m::add(arg0, arg1) }

fn main() { { let _ = add(1, 2); }; }
