#[prelude_import]
use ::std::prelude::rust_2015::*;
#[macro_use]
extern crate std;
// pretty-compare-only
// pretty-mode:hir
// pp-exact:hir-pretty-loop.pp

fn foo() { loop { break; } }
