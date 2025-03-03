#[prelude_import]
use ::std::prelude::rust_2015::*;
#[macro_use]
extern crate std;
//@ pretty-compare-only
//@ pretty-mode:hir
//@ pp-exact:hir-pretty-attr.pp

#[attr="Repr([ReprC, ReprPacked(Align(4 bytes)), ReprTransparent])")]
struct Example {
}
