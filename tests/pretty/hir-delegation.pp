//@ pretty-compare-only
//@ pretty-mode:hir
//@ pp-exact:hir-delegation.pp

#![allow(incomplete_features)]
#![feature(fn_delegation)]
extern crate std;
#[prelude_import]
use ::std::prelude::rust_2015::*;

fn b<C>(e: C) { }

trait G {
    #[attr = Inline(Hint)]
    fn b(arg0: _) -> _ { b({ }) }
}

mod m {
    fn add(a: u32, b: u32) -> u32 { a + b }
}

#[attr = Inline(Hint)]
fn add(arg0: _, arg1: _) -> _ { m::add(arg0, arg1) }

fn main() { { let _ = add(1, 2); }; }
