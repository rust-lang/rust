// Test that Cell is considered invariant with respect to its
// type.

// revisions: base nll
// ignore-compare-mode-nll
//[nll] compile-flags: -Z borrowck=mir

use std::cell::Cell;

struct Foo<'a> {
    x: Cell<Option<&'a isize>>,
}

fn use_<'short,'long>(c: Foo<'short>,
                      s: &'short isize,
                      l: &'long isize,
                      _where:Option<&'short &'long ()>) {
    let _: Foo<'long> = c;
    //[base]~^ ERROR E0623
    //[nll]~^^ ERROR lifetime may not live long enough
}

fn main() {
}
