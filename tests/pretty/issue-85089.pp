#[attr = MacroUse {arguments: UseAll}]
extern crate std;
#[prelude_import]
use ::std::prelude::rust_2015::*;
// Test to print lifetimes on HIR pretty-printing.

//@ pretty-compare-only
//@ pretty-mode:hir
//@ pp-exact:issue-85089.pp

trait A<'x> { }
trait B<'x> { }

struct Foo<'b> {
    bar: &'b dyn for<'a> A<'a>,
}

impl <'a> B<'a> for dyn for<'b> A<'b> { }

impl <'a> A<'a> for Foo<'a> { }
