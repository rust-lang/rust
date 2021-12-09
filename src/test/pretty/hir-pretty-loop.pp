#[prelude_import]
use ::std::prelude::rust_2015::*;
#[macro_use]
extern crate std;
// pretty-compare-only
// pretty-mode:hir
// pp-exact:hir-pretty-loop.pp

pub fn foo() { loop { break; } }
