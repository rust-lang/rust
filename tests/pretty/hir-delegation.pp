#![attr = LintAttributes([LintAttribute {kind: Allow, attr_style: Inner,
lint_instances: [incomplete_features]}])]
#![attr = Feature([fn_delegation#0])]
extern crate std;
#[attr = PreludeImport]
use ::std::prelude::rust_2015::*;
//@ pretty-compare-only
//@ pretty-mode:hir
//@ pp-exact:hir-delegation.pp


fn b<C>(e: C) { }

trait G {
    #[attr = Inline(Hint)]
    fn b<C>(arg0: _) -> _ { b::<C>({ }) }
}

mod m {
    fn add(a: u32, b: u32) -> u32 { a + b }
}

#[attr = Inline(Hint)]
fn add(arg0: _, arg1: _) -> _ { m::add(arg0, arg1) }

fn main() { { let _ = add(1, 2); }; }
