// Checks that regions which appear in a trait object type are
// observed by the variance inference algorithm (and hence
// `TOption` is contavariant w/r/t `'a` and not bivariant).
//
// Issue #18262.

#![feature(rustc_attrs)]

use std::mem;

trait T { fn foo(&self); }

#[rustc_variance]
struct TOption<'a> { //~ ERROR ['a: +]
    v: Option<Box<dyn T + 'a>>,
}

fn main() { }
