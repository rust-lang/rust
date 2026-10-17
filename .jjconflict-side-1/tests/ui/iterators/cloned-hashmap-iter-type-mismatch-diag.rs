//! Regression test for <https://github.com/rust-lang/rust/issues/33941>.
//! Test iterator type mismatch prints pretty error message.
//! This used to emit many duplicated, unhelpful error messages.
//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver
//@ compile-flags: -Zdeduplicate-diagnostics=yes

use std::collections::HashMap;

fn main() {
    for _ in HashMap::new().iter().cloned() {}
    //[current]~^ ERROR expected `Iter<'_, _, _>` to be an iterator that yields `&_`, but it yields `(&_, &_)`
    //[current]~| ERROR expected `Iter<'_, _, _>` to be an iterator that yields `&_`, but it yields `(&_, &_)`
    //[next]~^^^ ERROR: expected `Iter<'_, _, _>` to be an iterator that yields `&_`, but it yields `(&_, &_)`
    //[next]~| ERROR: expected `Iter<'_, _, _>` to be an iterator that yields `&_`, but it yields `(&_, &_)`
}
