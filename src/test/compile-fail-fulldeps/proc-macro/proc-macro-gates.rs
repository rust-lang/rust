// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// aux-build:proc-macro-gates.rs
// gate-test-proc_macro_non_items
// gate-test-proc_macro_path_invoc
// gate-test-proc_macro_mod line
// gate-test-proc_macro_expr
// gate-test-proc_macro_mod

#![feature(proc_macro, stmt_expr_attributes)]

extern crate proc_macro_gates as foo;

use foo::*;

#[foo::a] //~ ERROR: paths of length greater than one
fn _test() {}

#[a] //~ ERROR: custom attributes cannot be applied to modules
mod _test2 {}

#[a = y] //~ ERROR: must only be followed by a delimiter token
fn _test3() {}

#[a = ] //~ ERROR: must only be followed by a delimiter token
fn _test4() {}

#[a () = ] //~ ERROR: must only be followed by a delimiter token
fn _test5() {}

fn main() {
    #[a] //~ ERROR: custom attributes cannot be applied to statements
    let _x = 2;
    let _x = #[a] 2;
    //~^ ERROR: custom attributes cannot be applied to expressions

    let _x: m!(u32) = 3;
    //~^ ERROR: procedural macros cannot be expanded to types
    if let m!(Some(_x)) = Some(3) {
    //~^ ERROR: procedural macros cannot be expanded to patterns
    }
    let _x = m!(3);
    //~^ ERROR: procedural macros cannot be expanded to expressions
    m!(let _x = 3;);
    //~^ ERROR: procedural macros cannot be expanded to statements
}
