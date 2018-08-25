// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// As documented in Issue #45983, this test is evaluating the quality
// of our diagnostics on erroneous code using higher-ranked closures.
//
// However, as documented on Issue #53026, this test also became a
// prime example of our need to test the NLL migration mode
// *separately* from the existing test suites that focus solely on
// AST-borrwock and NLL.

// revisions: ast migrate nll

// Since we are testing nll (and migration) explicitly as a separate
// revisions, dont worry about the --compare-mode=nll on this test.

// ignore-compare-mode-nll

//[ast]compile-flags: -Z borrowck=ast
//[migrate]compile-flags: -Z borrowck=migrate -Z two-phase-borrows
//[nll]compile-flags: -Z borrowck=mir -Z two-phase-borrows

fn give_any<F: for<'r> FnOnce(&'r ())>(f: F) {
    f(&());
}

fn main() {
    let x = None;
    give_any(|y| x = Some(y));
    //[ast]~^ ERROR borrowed data cannot be stored outside of its closure
    //[migrate]~^^ ERROR borrowed data cannot be stored outside of its closure
    //[nll]~^^^ WARN not reporting region error due to nll
    //[nll]~| ERROR borrowed data escapes outside of closure
    //[nll]~| ERROR cannot assign to `x`, as it is not declared as mutable
}
