#[prelude_import]
use ::std::prelude::rust_2015::*;
#[attr = MacroUse {arguments: UseAll}]
extern crate std;
//@ pretty-compare-only
//@ pretty-mode:hir
//@ pp-exact:hir-pretty-loop.pp

fn foo() { loop { break; } }
