extern crate std;
#[attr = PreludeImport]
use ::std::prelude::rust_2015::*;
//@ pretty-compare-only
//@ pretty-mode:hir
//@ pp-exact:hir-pretty-attr.pp

#[attr = Repr {reprs: [ReprC, ReprPacked(Align(4 bytes)), ReprTransparent]}]
struct Example {
}
