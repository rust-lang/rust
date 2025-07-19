#[prelude_import]
use ::std::prelude::rust_2015::*;
#[attr = MacroUse {arguments: UseAll}]
extern crate std;
//@ pretty-compare-only
//@ pretty-mode:hir
//@ pp-exact:hir-pretty-attr.pp

#[attr = Repr {reprs: [ReprC, ReprPacked(Align(4 bytes)), ReprTransparent]}]
struct Example {
}
